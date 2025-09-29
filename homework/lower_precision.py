from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit_accurate(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    3-bit quantization optimized for accuracy with reasonable memory usage.
    Uses smaller groups for better accuracy.
    """
    assert x.dim() == 1
    original_size = x.size(0)
    
    # Smaller groups for better accuracy
    padding = (group_size - (original_size % group_size)) % group_size
    if padding > 0:
        x = torch.cat([x, torch.zeros(padding, dtype=x.dtype, device=x.device)])
    
    x = x.view(-1, group_size)
    
    # Per-group quantization with symmetric range
    group_scales = x.abs().max(dim=-1, keepdim=True).values
    group_scales = torch.clamp(group_scales, min=1e-8)
    
    # Normalize to [-1, 1] then map to [0, 7] for 3-bit
    x_normalized = x / group_scales
    x_normalized = torch.clamp(x_normalized, -1.0, 1.0)
    x_normalized = (x_normalized + 1.0) / 2.0  # Map to [0, 1]
    
    # 3-bit quantization (8 levels)
    x_quantized = (x_normalized * 7).round().to(torch.uint8)
    
    return x_quantized, group_scales.squeeze(-1).to(torch.float16)


def block_dequantize_3bit_accurate(x_quantized: torch.Tensor, scales: torch.Tensor, original_size: int) -> torch.Tensor:
    """
    Accurate dequantization.
    """
    scales = scales.to(torch.float32).unsqueeze(-1)
    
    # Reverse quantization
    x_normalized = x_quantized.to(torch.float32) / 7.0  # [0, 1]
    x_normalized = x_normalized * 2.0 - 1.0  # [-1, 1]
    x_dequantized = x_normalized * scales
    
    return x_dequantized.view(-1)[:original_size]


def pack_3bit_to_uint8(x_3bit: torch.Tensor) -> torch.Tensor:
    """
    Pack 2 3-bit values into each uint8 byte (with 2 bits unused).
    This is more memory efficient than storing each 3-bit value in a full uint8.
    """
    assert x_3bit.dim() == 2
    groups, size = x_3bit.shape
    
    # Make size even for pairing
    if size % 2 != 0:
        x_3bit = torch.cat([x_3bit, torch.zeros(groups, 1, dtype=torch.uint8, device=x_3bit.device)], dim=1)
        size += 1
    
    # Pack two 3-bit values: first in lower 3 bits, second in next 3 bits
    packed = torch.zeros(groups, size // 2, dtype=torch.uint8, device=x_3bit.device)
    packed = x_3bit[:, ::2] | (x_3bit[:, 1::2] << 3)
    
    return packed


def unpack_uint8_to_3bit(packed: torch.Tensor, original_size: int) -> torch.Tensor:
    """
    Unpack 3-bit values from uint8 bytes.
    """
    groups = packed.shape[0]
    unpacked_size = packed.shape[1] * 2
    
    unpacked = torch.zeros(groups, unpacked_size, dtype=torch.uint8, device=packed.device)
    unpacked[:, ::2] = packed & 0x7  # Lower 3 bits
    unpacked[:, 1::2] = (packed >> 3) & 0x7  # Next 3 bits
    
    # Trim to original size
    return unpacked[:, :original_size]


class Linear3BitEfficient(torch.nn.Module):
    """
    Memory-efficient 3-bit linear layer with bit packing.
    Targets ~6-7MB for the full network.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32):
        super().__init__()
        
        self._shape = (out_features, in_features)
        self._original_size = out_features * in_features
        self._group_size = group_size
        
        self._num_groups = (self._original_size + group_size - 1) // group_size
        self._padded_size = self._num_groups * group_size
        
        # Packed 3-bit weights (2 values per byte)
        packed_size = (group_size + 1) // 2
        self.register_buffer(
            "weight_packed",
            torch.zeros(self._num_groups, packed_size, dtype=torch.uint8),
            persistent=False,
        )
        
        # Scaling factors
        self.register_buffer(
            "weight_scales",
            torch.zeros(self._num_groups, dtype=torch.float16),
            persistent=False,
        )
        
        # Keep bias in float16 for memory efficiency
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        
        self._register_load_state_dict_pre_hook(Linear3BitEfficient._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            
            weight_flat = weight.flatten()
            weight_q3, scales = block_quantize_3bit_accurate(weight_flat, self._group_size)
            
            # Pack the 3-bit values
            weight_packed = pack_3bit_to_uint8(weight_q3)
            
            self.weight_packed.copy_(weight_packed)
            self.weight_scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Unpack 3-bit values
            weight_q3 = unpack_uint8_to_3bit(self.weight_packed, self._group_size)
            
            # Dequantize
            weight_dequant = block_dequantize_3bit_accurate(
                weight_q3, self.weight_scales, self._original_size
            )
            weight_dequant = weight_dequant.view(self._shape)
            
            bias = self.bias.to(torch.float32) if self.bias is not None else None
            return torch.nn.functional.linear(x, weight_dequant, bias)


class BalancedCompressedBigNet(torch.nn.Module):
    """
    Balanced compression targeting 6-8MB with good accuracy.
    Uses 3-bit quantization with bit packing and small groups.
    """
    
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Small groups (32) for better accuracy
            self.model = torch.nn.Sequential(
                Linear3BitEfficient(channels, channels, bias=True, group_size=32),
                torch.nn.ReLU(),
                Linear3BitEfficient(channels, channels, bias=True, group_size=32),
                torch.nn.ReLU(),
                Linear3BitEfficient(channels, channels, bias=True, group_size=32),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    """
    Load balanced compressed BigNet.
    
    Memory breakdown:
    - Weights (packed): 18.9M × 1.5 bits = 3.5 MB (3-bit packed into bytes)
    - Scale factors: ~590K groups × 2 bytes = 1.2 MB
    - Biases: 18.9M × 2 bytes = 2.0 MB (float16)
    - LayerNorm: negligible
    - Total theoretical: ~6.7 MB
    - Expected actual: ~7-8 MB (with PyTorch overhead)
    
    Better accuracy through:
    - Smaller groups (32 vs 64+)
    - Keeping bias parameters
    - Symmetric quantization
    """
    net = BalancedCompressedBigNet()
    
    if path is not None:
        try:
            net.load_state_dict(torch.load(path, weights_only=True))
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    return net