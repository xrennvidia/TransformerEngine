/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/transpose.h>

#include <algorithm>

#include "../util/rtc.h"
#include "../util/string.h"
#include "../utils.cuh"
#include "cast_transpose.h"

namespace transformer_engine::detail {

namespace {

// String with RTC kernel implementation
#include "string_code_transpose_rtc_cast_transpose_cu.h"

// Hard-coded kernel parameters
using CType = float;
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = THREADS_PER_WARP * warps_per_tile;

/* Performance heuristics for optimized kernel parameters */
struct KernelConfig {
  /** Vector load size */
  size_t load_size = 0;
  /** Vector store size to transposed output */
  size_t store_size = 0;

  /* Whether config is valid */
  bool valid = false;
  /* Number of CUDA blocks */
  size_t num_blocks = 0;

  /* Number of active SMs */
  size_t active_sm_count = 0;
  /* Elements per L1 cache load */
  size_t elements_per_load = 0;
  /* Elements per L1 cache store to cast output*/
  size_t elements_per_store_c = 0;
  /* Elements per L1 cache store to transposed output */
  size_t elements_per_store_t = 0;

  KernelConfig(size_t row_length, size_t num_rows, size_t itype_size, size_t otype_size,
               size_t load_size_, size_t store_size_, size_t sm_count)
      : load_size{load_size_}, store_size{store_size_} {
    // Check that tiles are correctly aligned
    constexpr size_t cache_line_size = 128;
    if (load_size % itype_size != 0 || store_size % otype_size != 0 ||
        cache_line_size % itype_size != 0 || cache_line_size % otype_size != 0) {
      return;
    }
    const size_t row_tile_elements = load_size * THREADS_PER_WARP / itype_size;
    const size_t col_tile_elements = store_size * THREADS_PER_WARP / otype_size;
    valid = (row_length % row_tile_elements == 0 && num_rows % col_tile_elements == 0);
    if (!valid) {
      return;
    }

    // Number of CUDA blocks
    num_blocks = (row_length / row_tile_elements) * (num_rows / col_tile_elements);

    // Parameters for performance model
    constexpr size_t warps_per_sm = 16;  // Rough estimate for saturated SMs
    active_sm_count = std::min(DIVUP(num_blocks * warps_per_tile, warps_per_sm), sm_count);
    elements_per_load = (std::min(cache_line_size, row_tile_elements * itype_size) / itype_size);
    elements_per_store_c = (std::min(cache_line_size, row_tile_elements * otype_size) / otype_size);
    elements_per_store_t = (std::min(cache_line_size, col_tile_elements * otype_size) / otype_size);
  }

  /* Compare by estimated cost */
  bool operator<(const KernelConfig &other) const {
    if (this->valid && other.valid) {
      // cost ~ (1/elements_per_load
      //         + 1/elements_per_store_c
      //         + 1/elements_per_store_t) / active_sms
      // Note: Integer arithmetic ensures stable ordering
      const auto &l1 = this->elements_per_load;
      const auto &sc1 = this->elements_per_store_c;
      const auto &st1 = this->elements_per_store_t;
      const auto &p1 = this->active_sm_count;
      const auto &l2 = other.elements_per_load;
      const auto &sc2 = other.elements_per_store_c;
      const auto &st2 = other.elements_per_store_t;
      const auto &p2 = other.active_sm_count;
      const auto scale = l1 * sc1 * st1 * p1 * l2 * sc2 * st2 * p2;
      const auto cost1 = (scale / l1 + scale / sc1 + scale / st1) / p1;
      const auto cost2 = (scale / l2 + scale / sc2 + scale / st2) / p2;
      return cost1 < cost2;
    } else {
      return this->valid && !other.valid;
    }
  }
};

template <size_t load_size, size_t store_size, typename IType, typename OType>
__global__ void __launch_bounds__(block_size) cast_transpose_general_kernel(
    const IType *__restrict__ const input, const CType *__restrict__ const noop,
    OType *__restrict__ const output_c, OType *__restrict__ const output_t,
    const CType *__restrict__ const scale_ptr, CType *__restrict__ const amax_ptr,
    CType *__restrict__ const scale_inv_ptr, const size_t row_length, const size_t num_rows) {
  if (noop != nullptr && noop[0] == 1.0f) return;

  // Vectorized load/store sizes
  constexpr size_t nvec_in = load_size / sizeof(IType);
  constexpr size_t nvec_out = store_size / sizeof(OType);
  using IVec = Vec<IType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr size_t bdimx = THREADS_PER_WARP;
  constexpr size_t bdimy = warps_per_tile;
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;
  const size_t bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Position of tile within tensor
  const size_t num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
  const size_t tile_id_m = bid % num_tiles_m;
  const size_t tile_id_n = bid / num_tiles_m;
  const size_t tile_row = tile_id_m * tile_dim_m;
  const size_t tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr size_t num_iterations = THREADS_PER_WARP / warps_per_tile;

  // FP8 factors
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType amax = 0;

  // Load input and store to registers
  // Note: Each thread loads num_iterations subtiles, computes amax,
  // casts type, and transposes in registers.
  OVecT local_output_t[nvec_in][num_iterations];
#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * bdimy;
    const size_t j1 = tidx;
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = tile_row + i1 * nvec_out + i2;
      const size_t col = tile_col + j1 * nvec_in;
      if (row < num_rows) {
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            const CType in = input[row * row_length + col + j2];
            const OType out = OType(in * scale);
            __builtin_assume(amax >= 0);
            amax = fmaxf(fabsf(in), amax);
            output_c[row * row_length + col + j2] = out;
            local_output_t[j2][iter].data.elt[i2] = out;
          }
        }
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVecT shared_output_t[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
      const size_t j1 = tidx;
      shared_output_t[j1][i1] = local_output_t[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidx;
      const size_t j1 = tidy + iter * bdimy;
      const size_t row = tile_row + i1 * nvec_out;
      const size_t col = tile_col + j1 * nvec_in + j2;
      if (col < row_length) {
#pragma unroll
        for (size_t i2 = 0; i2 < nvec_out; ++i2) {
          if (row + i2 < num_rows) {
            output_t[col * num_rows + row + i2] = shared_output_t[j1][i1].data.elt[i2];
          }
        }
      }
    }
    __syncthreads();
  }

  // Reduce amax over block
  if (amax_ptr != nullptr) {
    amax = reduce_max<warps_per_tile>(amax, tidy);
    if (threadIdx.x == 0) {
      static_assert(std::is_same<CType, float>::value);
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv_ptr != nullptr) {
    reciprocal<CType>(scale_inv_ptr, scale);
  }
}

}  // namespace

void cast_transpose(const Tensor &input, const Tensor &noop, Tensor *output_, cudaStream_t stream) {
  Tensor &output = *output_;

  CheckNoopTensor(noop, "cast_transpose_noop");
  CheckInputTensor(input, "cast_transpose_input");
  CheckOutputTensor(output, "cast_transpose_output");

  // Check that inputs and outputs are available
  NVTE_CHECK(input.has_data(), "Input is not allocated");
  NVTE_CHECK(output.has_data(), "Output rowwise data is not allocated");
  NVTE_CHECK(output.has_columnwise_data(), "Output columnwise is not allocated");

  // Flatten tensor to 2D
  NVTE_CHECK(input.data.shape == output.data.shape,
             "Input and output shapes do not match (input=", input.data.shape,
             ", output=", output.data.shape);
  const size_t row_length = input.flat_last_dim();
  const size_t num_rows = input.flat_first_dim();
  NVTE_CHECK(output.flat_first_dim() == num_rows && output.flat_last_dim() == row_length,
             "Invalid output dimensions (expected ", std::vector<size_t>{num_rows, row_length},
             ", got ", std::vector<size_t>{output.flat_first_dim(), output.flat_last_dim()}, ")");

  // Check that cast and transposed output data matches
  NVTE_CHECK(output.data.dtype == output.columnwise_data.dtype,
             "Cast and transposed output types must match.");
  NVTE_CHECK(output.scale_inv.dptr == output.columnwise_scale_inv.dptr,
             "Cast and transposed outputs need to share scale-inverse tensor.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output.dtype(), OutputType,
          if (is_tensor_scaling(output.scaling_mode)) {
            // delayed scaling and current scaling are two variants of per-tensor scaling

            constexpr const char *itype_name = TypeInfo<InputType>::name;
            constexpr const char *otype_name = TypeInfo<OutputType>::name;
            constexpr size_t itype_size = sizeof(InputType);
            constexpr size_t otype_size = sizeof(OutputType);

            // Choose between runtime-compiled or statically-compiled kernel
            const bool aligned =
                (row_length % THREADS_PER_WARP == 0 && num_rows % THREADS_PER_WARP == 0);
            if (aligned && rtc::is_enabled()) {  // Runtime-compiled tuned kernel
              // Pick kernel config
              std::vector<KernelConfig> kernel_configs;
              kernel_configs.reserve(16);
              const size_t sm_count = static_cast<size_t>(cuda::sm_count());
              auto add_config = [&](size_t load_size, size_t store_size) {
                kernel_configs.emplace_back(row_length, num_rows, itype_size, otype_size, load_size,
                                            store_size, sm_count);
              };
              add_config(8, 8);
              add_config(4, 8);
              add_config(8, 4);
              add_config(4, 4);
              add_config(2, 8);
              add_config(8, 2);
              add_config(2, 4);
              add_config(4, 2);
              add_config(2, 2);
              add_config(1, 8);
              add_config(8, 1);
              add_config(1, 4);
              add_config(4, 1);
              add_config(1, 2);
              add_config(2, 1);
              add_config(1, 1);
              const auto &kernel_config =
                  *std::min_element(kernel_configs.begin(), kernel_configs.end());
              NVTE_CHECK(kernel_config.valid, "invalid kernel config");
              const size_t load_size = kernel_config.load_size;
              const size_t store_size = kernel_config.store_size;
              const size_t num_blocks = kernel_config.num_blocks;

              // Compile NVRTC kernel if needed and launch
              auto &rtc_manager = rtc::KernelManager::instance();
              const std::string kernel_label = concat_strings(
                  "cast_transpose"
                  ",itype=",
                  itype_name, ",otype=", otype_name, ",load_size=", load_size,
                  ",store_size=", store_size);
              if (!rtc_manager.is_compiled(kernel_label)) {
                std::string code = string_code_transpose_rtc_cast_transpose_cu;
                code = regex_replace(code, "__ITYPE__", itype_name);
                code = regex_replace(code, "__OTYPE__", otype_name);
                code = regex_replace(code, "__LOAD_SIZE__", load_size);
                code = regex_replace(code, "__STORE_SIZE__", store_size);
                code = regex_replace(code, "__WARPS_PER_TILE__", warps_per_tile);
                code = regex_replace(code, "__BLOCK_SIZE__", block_size);
                rtc_manager.compile(kernel_label, "cast_transpose_optimized_kernel", code,
                                    "transformer_engine/common/transpose/rtc/cast_transpose.cu");
              }
              rtc_manager.launch(kernel_label, num_blocks, block_size, 0, stream,
                                 static_cast<const InputType *>(input.data.dptr),
                                 reinterpret_cast<const CType *>(noop.data.dptr),
                                 static_cast<OutputType *>(output.data.dptr),
                                 static_cast<OutputType *>(output.columnwise_data.dptr),
                                 static_cast<const CType *>(output.scale.dptr),
                                 static_cast<CType *>(output.amax.dptr),
                                 static_cast<CType *>(output.scale_inv.dptr), row_length, num_rows);
            } else {  // Statically-compiled general kernel
              constexpr size_t load_size = 4;
              constexpr size_t store_size = 4;
              constexpr size_t row_tile_size = load_size / itype_size * THREADS_PER_WARP;
              constexpr size_t col_tile_size = store_size / otype_size * THREADS_PER_WARP;
              const int num_blocks =
                  (DIVUP(row_length, row_tile_size) * DIVUP(num_rows, col_tile_size));

              cast_transpose_general_kernel<load_size, store_size, InputType, OutputType>
                  <<<num_blocks, block_size, 0, stream>>>(
                      static_cast<const InputType *>(input.data.dptr),
                      reinterpret_cast<const CType *>(noop.data.dptr),
                      static_cast<OutputType *>(output.data.dptr),
                      static_cast<OutputType *>(output.columnwise_data.dptr),
                      static_cast<const CType *>(output.scale.dptr),
                      static_cast<CType *>(output.amax.dptr),
                      static_cast<CType *>(output.scale_inv.dptr), row_length, num_rows);
            }
          } else {
            NVTE_ERROR("Not implemented scaling mode: ", to_string(output.scaling_mode));
          });  // NOLINT(*)
  );           // NOLINT(*)
}

}  // namespace transformer_engine::detail

void nvte_cast_transpose(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose);
  using namespace transformer_engine;
  auto noop = Tensor();
  transformer_engine::detail::cast_transpose(*convertNVTETensorCheck(input), noop,
                                             convertNVTETensor(output), stream);
}

void nvte_cast_transpose_with_noop(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_with_noop);
  using namespace transformer_engine;
  transformer_engine::detail::cast_transpose(*convertNVTETensorCheck(input),
                                             *convertNVTETensorCheck(noop),
                                             convertNVTETensor(output), stream);
}
