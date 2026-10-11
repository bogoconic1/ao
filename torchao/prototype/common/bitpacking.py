"""
Pack low-bit unsigned integer values into wider integer containers, and back.

Each container holds k = container_bits // element_bits values. Value j of a
group sits at bit offset j * element_bits, least significant bits first; any
spare bits at the top of the container are zero. For example, 3-bit values in
an int16 container (k = 5, one spare bit):

    bit: 15 | 14..12 | 11..9 | 8..6 | 5..3 | 2..0
          0 |   v4   |   v3  |  v2  |  v1  |  v0

Packing happens along a single dim (dim=1 packs adjacent values within a row,
dim=0 within a column). If that dim is not a multiple of k, the last group is
zero-padded; pass the original length as `size` to `unpack` to trim it.

torch.uint16 is not a supported container because bit shifts are not
implemented for it on CPU or CUDA; use torch.int16 instead. Signed containers
are safe: packing only shifts and ORs, and unpacking masks after every shift.
"""

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["pack", "unpack"]

_CONTAINER_BITS = {torch.uint8: 8, torch.int16: 16, torch.int32: 32}
_INPUT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _container_bits(dtype: torch.dtype) -> int:
    if dtype == torch.uint16:
        raise ValueError(
            "torch.uint16 is not supported as a container because bit shifts "
            "are not implemented for it; use torch.int16 instead"
        )
    if dtype not in _CONTAINER_BITS:
        raise ValueError(
            f"Unsupported container dtype {dtype}, "
            f"expected one of {list(_CONTAINER_BITS)}"
        )
    return _CONTAINER_BITS[dtype]


def _check_element_bits(element_bits: int, container_bits: int) -> None:
    if not 1 <= element_bits <= min(8, container_bits):
        raise ValueError(
            f"element_bits must be between 1 and {min(8, container_bits)}, "
            f"got {element_bits}"
        )


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("Cannot pack a 0-dimensional tensor")
    if not -ndim <= dim < ndim:
        raise ValueError(f"dim {dim} is out of range for a {ndim}-D tensor")
    return dim % ndim


def pack(
    data: torch.Tensor,
    element_bits: int,
    container_dtype: torch.dtype = torch.uint8,
    dim: int = -1,
) -> torch.Tensor:
    """
    Packs `data` along `dim` into a tensor of `container_dtype`.

    Values are treated as unsigned and masked to their low `element_bits`
    bits, so out-of-range bits never leak into neighbouring values. The packed
    dim shrinks to ceil(n / k); the result is contiguous.
    """
    if data.dtype not in _INPUT_DTYPES:
        raise ValueError(
            f"Unsupported input dtype {data.dtype}, "
            f"expected one of {list(_INPUT_DTYPES)}"
        )
    container_bits = _container_bits(container_dtype)
    _check_element_bits(element_bits, container_bits)
    dim = _normalize_dim(dim, data.dim())

    k = container_bits // element_bits
    mask = (1 << element_bits) - 1

    # cast before masking so the low bits of wider or negative inputs survive
    x = data.movedim(dim, -1).to(container_dtype) & mask
    n = x.shape[-1]
    pad = (-n) % k
    if pad:
        x = F.pad(x, (0, pad))
    x = x.unflatten(-1, ((n + pad) // k, k))

    packed = x[..., 0]
    for j in range(1, k):
        packed = packed | (x[..., j] << (j * element_bits))
    return packed.movedim(-1, dim).contiguous()


def unpack(
    packed: torch.Tensor,
    element_bits: int,
    dim: int = -1,
    size: Optional[int] = None,
) -> torch.Tensor:
    """
    Unpacks a tensor produced by `pack` back into uint8 values.

    The container dtype is taken from `packed.dtype`. Without `size` the
    unpacked dim is packed.shape[dim] * k, including any zero padding; pass the
    original length as `size` to trim the padding.
    """
    container_bits = _container_bits(packed.dtype)
    _check_element_bits(element_bits, container_bits)
    dim = _normalize_dim(dim, packed.dim())

    k = container_bits // element_bits
    mask = (1 << element_bits) - 1
    capacity = packed.shape[dim] * k
    if size is not None and not 0 <= size <= capacity:
        raise ValueError(f"size must be between 0 and {capacity}, got {size}")

    p = packed.movedim(dim, -1)
    # the mask also clears sign bits that >> copies in for signed containers
    parts = [(p >> (j * element_bits)) & mask for j in range(k)]
    out = torch.stack(parts, dim=-1).flatten(-2)
    if size is not None:
        out = out[..., :size]
    return out.to(torch.uint8).movedim(-1, dim).contiguous()
