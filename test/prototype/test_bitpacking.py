import pytest

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torchao.prototype.common.bitpacking import pack, unpack

torch.manual_seed(0)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]

# (element_bits, container_dtype)
SPEC_FORMATS = [(4, torch.uint8), (3, torch.int16), (2, torch.uint8)]
EXTRA_FORMATS = [
    (4, torch.int16),  # top slot occupies the sign bit
    (8, torch.int32),  # 255 << 24 sets the sign bit
    (1, torch.int32),  # k = 32, the longest unrolled loop
    (8, torch.uint8),  # k = 1, nothing to pack together
]
ALL_FORMATS = SPEC_FORMATS + EXTRA_FORMATS

CONTAINER_BITS = {torch.uint8: 8, torch.int16: 16, torch.int32: 32}


def _values_per_container(element_bits, container_dtype):
    return CONTAINER_BITS[container_dtype] // element_bits


def _random_values(element_bits, shape, device):
    return torch.randint(0, 2**element_bits, shape, dtype=torch.uint8, device=device)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    # avoid dynamo cache limit issues
    torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("element_bits, container_dtype", ALL_FORMATS)
def test_roundtrip(element_bits, container_dtype, dim, device):
    # 160 is divisible by every k in ALL_FORMATS (1, 2, 4, 5, 32): no padding
    x = _random_values(element_bits, (160, 160), device)
    k = _values_per_container(element_bits, container_dtype)

    packed = pack(x, element_bits, container_dtype, dim=dim)

    expected_shape = list(x.shape)
    expected_shape[dim] //= k
    assert packed.dtype == container_dtype
    assert list(packed.shape) == expected_shape
    assert packed.device == x.device
    assert packed.is_contiguous()
    torch.testing.assert_close(unpack(packed, element_bits, dim=dim), x, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(7, 13), (1, 1)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("element_bits, container_dtype", ALL_FORMATS)
def test_padding(element_bits, container_dtype, dim, shape, device):
    x = _random_values(element_bits, shape, device)
    k = _values_per_container(element_bits, container_dtype)
    n = x.shape[dim]

    packed = pack(x, element_bits, container_dtype, dim=dim)
    assert packed.shape[dim] == -(-n // k)

    trimmed = unpack(packed, element_bits, dim=dim, size=n)
    torch.testing.assert_close(trimmed, x, atol=0, rtol=0)

    untrimmed = unpack(packed, element_bits, dim=dim)
    assert untrimmed.shape[dim] == packed.shape[dim] * k
    torch.testing.assert_close(untrimmed.narrow(dim, 0, n), x, atol=0, rtol=0)
    assert (untrimmed.narrow(dim, n, untrimmed.shape[dim] - n) == 0).all()


@pytest.mark.parametrize(
    "values, element_bits, container_dtype, expected",
    [
        ([3, 0, 2, 1], 2, torch.uint8, 0b01100011),
        ([1, 2], 4, torch.uint8, 0x21),
        ([1, 2, 3, 4, 5], 3, torch.int16, 1 | 2 << 3 | 3 << 6 | 4 << 9 | 5 << 12),
        ([0, 0, 0, 8], 4, torch.int16, -32768),
    ],
)
def test_bit_layout(values, element_bits, container_dtype, expected):
    x = torch.tensor([values], dtype=torch.uint8)
    packed = pack(x, element_bits, container_dtype)
    assert packed.tolist() == [[expected]]


def test_bit_layout_spare_bit_is_zero():
    x = torch.full((1, 5), 7, dtype=torch.uint8)
    packed = pack(x, 3, torch.int16)
    assert packed.item() == 0x7FFF


@pytest.mark.parametrize("element_bits, container_dtype", SPEC_FORMATS)
def test_rowwise_matches_colwise(element_bits, container_dtype):
    x = _random_values(element_bits, (12, 20), "cpu")
    colwise = pack(x, element_bits, container_dtype, dim=0)
    rowwise_of_transpose = pack(x.T, element_bits, container_dtype, dim=1)
    torch.testing.assert_close(colwise, rowwise_of_transpose.T, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((10,), 0),  # 1-D
        ((3, 10, 4), 1),  # 3-D, middle dim
        ((3, 10, 4), -3),  # 3-D, negative dim
        ((0, 5), 0),  # empty packed dim
        ((0, 5), 1),  # empty other dim
    ],
)
def test_other_shapes(shape, dim, device):
    x = _random_values(3, shape, device)
    packed = pack(x, 3, torch.int16, dim=dim)
    out = unpack(packed, 3, dim=dim, size=x.shape[dim])
    torch.testing.assert_close(out, x, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_input(device):
    x = _random_values(4, (8, 6), device).T
    assert not x.is_contiguous()
    packed = pack(x, 4, torch.uint8, dim=1)
    torch.testing.assert_close(unpack(packed, 4, dim=1), x, atol=0, rtol=0)


@pytest.mark.parametrize("input_dtype", [torch.int8, torch.int16, torch.int64])
def test_input_dtypes(input_dtype):
    x = _random_values(3, (4, 10), "cpu")
    packed = pack(x.to(input_dtype), 3, torch.int16)
    out = unpack(packed, 3, size=10)
    assert out.dtype == torch.uint8
    torch.testing.assert_close(out, x, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_masks_out_of_range(device):
    x = torch.tensor([[1, 0xFF, 2, 3]], dtype=torch.uint8, device=device)
    out = unpack(pack(x, 2, torch.uint8), 2)
    assert out.tolist() == [[1, 3, 2, 3]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize(
    "element_bits, container_dtype",
    SPEC_FORMATS + [(4, torch.int16), (8, torch.int32)],
)
def test_compile_fullgraph(element_bits, container_dtype, dim, device):
    x = _random_values(element_bits, (7, 13), device)
    n = x.shape[dim]
    counter = CompileCounterWithBackend("inductor")

    pack_c = torch.compile(
        lambda t: pack(t, element_bits, container_dtype, dim=dim),
        backend=counter,
        fullgraph=True,
        dynamic=False,
    )
    unpack_c = torch.compile(
        lambda t: unpack(t, element_bits, dim=dim, size=n),
        backend=counter,
        fullgraph=True,
        dynamic=False,
    )

    packed_c = pack_c(x)
    packed_e = pack(x, element_bits, container_dtype, dim=dim)
    torch.testing.assert_close(packed_c, packed_e, atol=0, rtol=0)

    out_c = unpack_c(packed_c)
    torch.testing.assert_close(out_c, x, atol=0, rtol=0)
    assert counter.frame_count == 2


def test_invalid_args():
    x = torch.zeros(2, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="int16"):
        pack(x, 3, torch.uint16)
    with pytest.raises(ValueError, match="container dtype"):
        pack(x, 4, torch.int64)
    with pytest.raises(ValueError, match="element_bits"):
        pack(x, 0)
    with pytest.raises(ValueError, match="element_bits"):
        pack(x, 9, torch.int16)
    with pytest.raises(ValueError, match="input dtype"):
        pack(x.float(), 4)
    with pytest.raises(ValueError, match="input dtype"):
        pack(x.bool(), 1)
    with pytest.raises(ValueError, match="out of range"):
        pack(x, 4, dim=2)
    with pytest.raises(ValueError, match="0-dimensional"):
        pack(torch.tensor(1, dtype=torch.uint8), 4)

    packed = pack(x, 4)
    with pytest.raises(ValueError, match="size"):
        unpack(packed, 4, size=5)
    with pytest.raises(ValueError, match="size"):
        unpack(packed, 4, size=-1)
    with pytest.raises(ValueError, match="container dtype"):
        unpack(packed.to(torch.int64), 4)
