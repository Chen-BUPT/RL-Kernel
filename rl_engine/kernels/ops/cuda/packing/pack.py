# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native CUDA fused masking + variable-length pack-and-pad op (issue #42).

Wraps the precompiled ``_C.pack_forward`` / ``_C.pack_backward`` kernels in a
``torch.autograd.Function``. Forward gathers the active rows of a dense
``[B, S, *tail]`` tensor (selected by a ``[B, S]`` mask) into a contiguous
``[Total_Active, *tail]`` tensor and returns per-row ``cu_seqlens``; backward
scatters the gradient back to the dense layout with zeros at inactive
positions.

Packing order is row-major over the flattened ``[B, S]`` grid, identical to
``NativePackOp`` / ``TritonPackOp``. The destination index (packed-row ->
source-row) and ``cu_seqlens`` are computed here with cheap torch ops; the
CUDA kernels only move the tail vectors, so the grid is launched over the
active rows only (no wasted blocks on masked-out rows).
"""

from __future__ import annotations

from typing import Tuple

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


def _dest_and_cu_seqlens(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (src_row [n_active] int64, cu_seqlens [B+1] int64).

    ``src_row[p]`` is the flattened source-row index of the p-th packed row.
    """
    flat = mask.reshape(-1).to(torch.bool)
    src_row = flat.nonzero(as_tuple=False).squeeze(-1).to(torch.int64).contiguous()

    per_row_active = mask.reshape(mask.shape[0], -1).to(torch.int64).sum(dim=1)
    cu_seqlens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device=mask.device)
    torch.cumsum(per_row_active, dim=0, out=cu_seqlens[1:])
    return src_row, cu_seqlens


class _PackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        lead = mask.dim()
        tail_shape = x.shape[lead:]
        n_rows = 1
        for s in mask.shape:
            n_rows *= int(s)
        T = 1
        for s in tail_shape:
            T *= int(s)

        src = x.reshape(n_rows, T).contiguous()
        src_row, cu_seqlens = _dest_and_cu_seqlens(mask)

        packed, cu_seqlens = _C.pack_forward(src, src_row, cu_seqlens)

        ctx.save_for_backward(src_row)
        ctx.n_rows = n_rows
        ctx.x_shape = tuple(x.shape)
        n_active = src_row.shape[0]
        out_tail = tuple(tail_shape)
        packed_out = packed.reshape(n_active, *out_tail) if out_tail else packed.reshape(n_active)
        return packed_out, cu_seqlens

    @staticmethod
    def backward(ctx, grad_packed: torch.Tensor, grad_cu_seqlens):
        (src_row,) = ctx.saved_tensors
        n_active = src_row.shape[0]
        if n_active == 0:
            grad_x = grad_packed.new_zeros(ctx.x_shape)
            return grad_x, None
        # Flatten grad to [n_active, T] matching the forward packed layout.
        grad_2d = grad_packed.reshape(n_active, -1).contiguous()
        grad_src = _C.pack_backward(grad_2d, src_row, ctx.n_rows)
        grad_x = grad_src.reshape(ctx.x_shape)
        return grad_x, None


class CudaPackOp:
    """Native CUDA fused masking + variable-length packing (pack-and-pad).

    Forward packs the active rows of ``x`` (selected by ``mask``) into a
    contiguous ``[Total_Active, *tail]`` tensor and returns the per-row
    ``cu_seqlens`` prefix-sum. Backward scatters the upstream gradient back to
    the original ``[*mask.shape, *tail]`` layout. Numerically identical to
    ``NativePackOp``.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "pack_forward"):
            raise RuntimeError(
                "Native CUDA pack kernel is not compiled. Rebuild with 'pip install -e .'."
            )
        logger.info("Successfully linked to precompiled _C.pack_forward kernel.")

    def __call__(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError(f"CudaPackOp requires a CUDA tensor, got device '{x.device}'.")
        if mask.dim() < 1:
            raise ValueError("mask must have at least one dimension.")
        if mask.shape != x.shape[: mask.dim()]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match the leading dims of "
                f"x.shape {tuple(x.shape)} (expected {tuple(x.shape[: mask.dim()])})."
            )
        return _PackFunction.apply(x, mask)
