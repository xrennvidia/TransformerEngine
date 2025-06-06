/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file recipe.h
 *  \brief Functions handling FP8 recipes.
 */

#ifndef TRANSFORMER_ENGINE_RECIPE_H_
#define TRANSFORMER_ENGINE_RECIPE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Update FP8 scaling factors with delayed scaling recipe.
 *
 * The amax history is rotated by -1 (e.g. the first entry shifts to
 * the last, the last entry shifts to the second to last) and the
 * first entry is set to zero. The scaling factor is estimated so the
 * FP8 tensor's maximum absolute value is
 * @f$ 2^{-\text{margin}} \text{max}_\text{fp8\_dtype} @f$.
 *
 *  \param[in] amax_history             History of maximum absolute values.
 *                                      Shape: [history_length, num_scales]
 *  \param[in] scale                    Scaling factor for casting to FP8. Shape: [num_scales]
 *  \param[out] updated_amax_history    Updated history of maximum absolute values.
 *                                      Shape: [history_length, num_scales]
 *  \param[out] updated_scale           Updated scaling factor for casting to FP8.
 *                                      Shape: [num_scales]
 *  \param[in] amax_compute_algo        Method to reduce amax history. Options are "max" and
 *                                      "most_recent".
 *  \param[in] fp8_dtype                FP8 datatype.
 *  \param[in] margin                   Scaling factor margin.
 *  \param[in] stream                   CUDA stream.
 */
void nvte_delayed_scaling_recipe_amax_and_scale_update(
    const NVTETensor amax_history, const NVTETensor scale, NVTETensor updated_amax_history,
    NVTETensor updated_scale, const char* amax_compute_algo, NVTEDType fp8_dtype, float margin,
    cudaStream_t stream);

/*! \brief Bulk-update FP8 scaling factors with delayed scaling recipe after amax reduction.
 *
 * Operations performed include, updating the most recent amax history
 * with the relevant segment of global reduction buffer if it's not 0,
 * rotating the amax history based on the rule below, and updating the
 * scales.
 *
 * The amax history is rotated by -1 (e.g. the first entry shifts to
 * the last, the last entry shifts to the second to last) and the
 * first entry is set to zero. The scaling factor is estimated so the
 * FP8 tensor's maximum absolute value is
 * @f$ 2^{-\text{margin}} \text{max}_\text{fp8\_dtype} @f$.
 *
 *  \param[in] amax_reduction_buffer    The contiguous buffer used for amax reduction.
 *                                      Shape: [num_scales * num_tensors]
 *  \param[in,out] amax_histories       List of amax histories of maximum absolute values.
 *                                      Shape: num_tensors x [history_length, num_scales]
 *  \param[in,out] scales               List of scaling factors for casting to FP8.
 *                                      Shape: num_tensors x [num_scales]
 *  \param[in] amax_compute_algo        Method to reduce amax history. Options are "max" and
 *                                      "most_recent".
 *  \param[in] fp8_dtype                FP8 datatype.
 *  \param[in] margin                   Scaling factor margin.
 *  \param[in] stream                   CUDA stream.
 */
void nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
    const NVTETensor amax_reduction_buffer, std::vector<NVTETensor> amax_histories,
    std::vector<NVTETensor> scales, const char* amax_compute_algo, NVTEDType fp8_dtype,
    float margin, cudaStream_t stream);

/*! \brief Compute an FP8 tensor's amax.
 *
 *  The amax (maximum absolute value) of the input tensor is computed
 *  and written to the amax buffer of the output tensor.
 *
 *  \param[in]     input            Input tensor. Must be unquantized.
 *  \param[in,out] output           Output tensor. Must be an FP8 tensor with per-tensor scaling.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_compute_amax(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Update an FP8 tensor's scale based on its amax.
 *
 *  This is only supported for FP8 tensors with per-tensor scaling.
 *  Options are primarily intended for FP8 current-scaling recipes.
 *
 *  \param[in,out] output           FP8 tensor with per-tensor scaling.
 *  \param[in]     config           Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_compute_scale_from_amax(NVTETensor output, const NVTEQuantizationConfig config,
                                  cudaStream_t stream);

void nvte_fp8_block_scaling_compute_partial_amax(const NVTETensor inp, NVTETensor amax, size_t h,
                                                 size_t w, size_t amax_stride_h,
                                                 size_t amax_stride_w, size_t start_offset,
                                                 size_t block_len, cudaStream_t stream);

void nvte_fp8_block_scaling_partial_cast(const NVTETensor inp, NVTETensor out,
                                         const NVTETensor scale, size_t h, size_t w,
                                         size_t scale_stride_h, size_t scale_stride_w,
                                         size_t start_offset, size_t block_len,
                                         const NVTEDType out_dtype, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RECIPE_H_
