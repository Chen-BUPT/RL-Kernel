// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// Fused masking + variable-length pack-and-pad CUDA kernels (issue #42).
//
// pack_forward: gather the active rows of a flattened [n_rows, T] tensor
//   (selected by a [n_rows] mask) into a contiguous [n_active, T] tensor, and
//   return per-row cu_seqlens (prefix-sum of per-row active counts).
// pack_backward: scatter a [n_active, T] gradient back to a zero-initialized
//   [n_rows, T] tensor at the active rows.
//
// The packing order is row-major over the flattened mask, identical to the
// NativePackOp / TritonPackOp reference. Destination indices are computed on
// the host with torch ops (cheap, [n_rows]); the kernels only move the
// T-wide tail vectors.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;

// One block handles one packed (active) row; threads stride over the T columns.
template <typename scalar_t>
__global__ void pack_gather_kernel(
    const scalar_t* __restrict__ src,   // [n_rows, T]
    scalar_t* __restrict__ dst,         // [n_active, T]
    const int64_t* __restrict__ src_row, // [n_active] -> source row for each packed row
    int64_t n_active,
    int64_t T) {
  const int64_t packed_row = blockIdx.x;
  if (packed_row >= n_active) {
    return;
  }
  const int64_t s = src_row[packed_row] * T;
  const int64_t d = packed_row * T;
  for (int64_t col = threadIdx.x; col < T; col += blockDim.x) {
    dst[d + col] = src[s + col];
  }
}

// Backward: scatter each packed row's gradient back to its source row.
template <typename scalar_t>
__global__ void pack_scatter_kernel(
    const scalar_t* __restrict__ grad_packed, // [n_active, T]
    scalar_t* __restrict__ grad_src,          // [n_rows, T], pre-zeroed
    const int64_t* __restrict__ src_row,      // [n_active]
    int64_t n_active,
    int64_t T) {
  const int64_t packed_row = blockIdx.x;
  if (packed_row >= n_active) {
    return;
  }
  const int64_t s = src_row[packed_row] * T;
  const int64_t d = packed_row * T;
  for (int64_t col = threadIdx.x; col < T; col += blockDim.x) {
    grad_src[s + col] = grad_packed[d + col];
  }
}

}  // namespace

// Returns {packed [n_active, T], cu_seqlens [B + 1]}.
// src: [n_rows, T] contiguous; mask: [B, S] (n_rows == B * S); src_row maps each
// packed row index to its source row (computed on the host by the caller).
std::vector<torch::Tensor> pack_forward(
    torch::Tensor src,
    torch::Tensor src_row,
    torch::Tensor cu_seqlens) {
  TORCH_CHECK(src.is_cuda(), "pack_forward: src must be a CUDA tensor");
  TORCH_CHECK(src.dim() == 2, "pack_forward: src must be 2-D [n_rows, T]");
  TORCH_CHECK(src.is_contiguous(), "pack_forward: src must be contiguous");
  TORCH_CHECK(src_row.is_cuda() && src_row.dtype() == torch::kInt64,
              "pack_forward: src_row must be an int64 CUDA tensor");

  const int64_t T = src.size(1);
  const int64_t n_active = src_row.size(0);

  auto packed = torch::empty({n_active, T}, src.options());

  if (n_active > 0) {
    const int threads = static_cast<int>(std::min<int64_t>(T, kThreadsPerBlock));
    const dim3 grid(static_cast<unsigned int>(n_active));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        src.scalar_type(), "pack_gather", [&] {
          pack_gather_kernel<scalar_t><<<grid, threads, 0, stream>>>(
              src.data_ptr<scalar_t>(),
              packed.data_ptr<scalar_t>(),
              src_row.data_ptr<int64_t>(),
              n_active, T);
        });
  }
  return {packed, cu_seqlens};
}

// Returns grad_src [n_rows, T] (zeros at inactive rows).
torch::Tensor pack_backward(
    torch::Tensor grad_packed,
    torch::Tensor src_row,
    int64_t n_rows) {
  TORCH_CHECK(grad_packed.is_cuda(), "pack_backward: grad_packed must be CUDA");
  TORCH_CHECK(grad_packed.dim() == 2, "pack_backward: grad_packed must be 2-D");
  TORCH_CHECK(src_row.is_cuda() && src_row.dtype() == torch::kInt64,
              "pack_backward: src_row must be an int64 CUDA tensor");

  const int64_t T = grad_packed.size(1);
  const int64_t n_active = src_row.size(0);

  auto grad_packed_c = grad_packed.contiguous();
  auto grad_src = torch::zeros({n_rows, T}, grad_packed_c.options());

  if (n_active > 0) {
    const int threads = static_cast<int>(std::min<int64_t>(T, kThreadsPerBlock));
    const dim3 grid(static_cast<unsigned int>(n_active));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_packed_c.scalar_type(), "pack_scatter", [&] {
          pack_scatter_kernel<scalar_t><<<grid, threads, 0, stream>>>(
              grad_packed_c.data_ptr<scalar_t>(),
              grad_src.data_ptr<scalar_t>(),
              src_row.data_ptr<int64_t>(),
              n_active, T);
        });
  }
  return grad_src;
}
