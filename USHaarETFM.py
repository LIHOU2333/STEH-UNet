import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from timm.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional
import math

# from efficient_kan import KAN
# from OURmodels.efficient_kan import KAN

from pytorch_wavelets import DWTForward
# from pytorch_wavelets import DWTInverse
from utils.CA import CoordAtt


# __all__ = ['KANLayer', 'KANBlock','UKAN_CBAM','D_ConvLayer','ConvLayer','PatchEmbed','DW_bn_relu','DWConv']
__all__ = ['SwinTransformerBlock','UTKAN',  'PatchEmbed', 'DropPath']

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#用于实现随机深度（Stochastic Depth），在训练过程中随机丢弃路径以增强模型的泛化能力
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


# 比vision_transformer.py多的部分
# window_partition：将特征图划分为不重叠的窗口。
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 比vision_transformer.py多的部分
# window_reverse：将划分后的窗口重新组合为完整的特征图。
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#将2D图像划分为补丁（Patch），并将其嵌入到高维空间。
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    # embed_dim 参数在 PatchEmbed 类中定义了每个补丁（patch）被映射到的嵌入向量的维度。
    # 具体来说，它决定了卷积层 self.proj 的输出通道数，即特征图经过卷积后的深度或通道数。
    def __init__(self, patch_size=4, in_c=320, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW] transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# 比vision_transformer.py多的部分
# 基于窗口的多头自注意力机制，支持相对位置偏置。
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#构成Swin Transformer的基本模块，包含自注意力和多层感知机。
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Layer Normalization before W-MSA and SW-MSA
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Layer Normalization before MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        #输入特征 x（维度 [B, L, C]，其中 B 是批次大小，L 是序列长度，C 是特征维度）
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size填充功能映射到窗口大小的倍数
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN Feed-forward network
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W

# 小波变换中的低频和高频分量
# 在小波变换（DWT, Discrete Wavelet Transform）中，图像被分解成多个不同分辨率的子带，每个子带对应不同的频率范围：
#
# 低频分量（LL）：表示图像的近似部分，类似于缩小版的原始图像，保留了图像的主要结构信息。
# 高频分量：分为三个方向的高频分量：
# LH（水平方向的高频分量）：捕获图像中的垂直边缘信息。
# HL（垂直方向的高频分量）：捕获图像中的水平边缘信息。
# HH（对角线方向的高频分量）：捕获图像中的对角线边缘信息。

# 小波变换下采样
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    # 如果你的输入张量有3个通道（如RGB图像），那么在小波变换后会有 3 * 4 = 12 个通道。
    # 接着通过 conv_bn_relu 后，你可以选择输出任何数量的通道（由 out_ch 参数决定）。
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class EnhancedTFusionV3(nn.Module):

    def __init__(self, t_low_dim, t_high_dim, out_dim):
        super().__init__()
        self.down_sample = nn.Sequential(

            nn.Conv2d(t_low_dim, t_low_dim // 4, 3, stride=2, padding=1),  # 减少通道
            nn.BatchNorm2d(t_low_dim // 4),
            nn.GELU(),
            nn.Conv2d(t_low_dim // 4, t_low_dim, 1)  # 恢复通道
        )
        self.conv_low = nn.Conv2d(t_low_dim, out_dim, 1)
        self.conv_high = nn.Conv2d(t_high_dim, out_dim, 1)

        self.ca = CoordAtt(inp=out_dim, oup=out_dim)
        self.res_conv = nn.Sequential(
            nn.Conv2d(t_high_dim, out_dim, 3, padding=1, groups=4),  # 分组卷积
            nn.BatchNorm2d(out_dim)
        ) if t_high_dim != out_dim else nn.Identity()
        self.dark_enhance = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid()  # 生成0-1的增强系数
        )

    def forward(self, t_low, t_high):
        t_low_down = self.down_sample(t_low)
        t_low_proj = self.conv_low(t_low_down)
        t_high_proj = self.conv_high(t_high)


        fused = t_low_proj + t_high_proj  # 先相加减少计算量
        att_fused = self.ca(fused)


        fused_feat = t_low_proj * att_fused + t_high_proj * (1 - att_fused)

        dark_mask = (t_high.mean(dim=1, keepdim=True) < 0.5).float()
        enhance_coeff = self.dark_enhance(fused_feat)
        fused_feat = fused_feat * (1 + enhance_coeff * dark_mask)
        return fused_feat + self.res_conv(t_high)

class UTKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=None, input_list=None, no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [256,320,512]
        if input_list is None:
            input_list = [32, 64, 256, 320, 512]
        self.input_list = input_list
        if depths is None:
            depths = [2, 2, 2]

        self.encode_layer1 = self._make_encode_layer(3, input_list[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = self._make_encode_layer(input_list[0], input_list[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = self._make_encode_layer(input_list[1], input_list[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_wt1 = Down_wt(input_list[0],input_list[0])
        self.down_wt2 = Down_wt(input_list[1], input_list[1])

        self.decode_layer4 = self._make_decode_layer(input_list[4], input_list[3])
        self.decode3 = self._make_decode_layer(input_list[3], input_list[2]) #用于上采样中的卷积
        self.decode_layer3 = self._make_decode_layer(input_list[2]*2, input_list[2]) #用于cat之后的卷积
        self.decode2 = self._make_decode_layer(input_list[2], out_channels=input_list[1])
        self.decode_layer2 = self._make_decode_layer(input_list[1]*2, out_channels=input_list[1])
        self.decode1 = self._make_decode_layer(input_list[1], input_list[0])
        self.decode_layer1 = self._make_decode_layer(input_list[0], input_list[0])

        # 输出层
        self.out_layer = nn.Conv2d(in_channels=input_list[0], out_channels=num_classes, kernel_size=1, padding=0,stride=1)

        self.fusion3 = EnhancedTFusionV3(
            t_low_dim=input_list[2],  # 低层特征维度128（对应编码器t3） input_list[2]
            t_high_dim=input_list[3],  # 高层特征维度256（对应解码器输出）input_list[3]
            out_dim=input_list[3]  # 输出维度256（与中间层匹配）embed_dims[1]
        )
        self.fusion2 = EnhancedTFusionV3(
            t_low_dim=input_list[1],  # 64
            t_high_dim=input_list[2],  # 128
            out_dim=input_list[2]  # 128
        )
        self.fusion1 = EnhancedTFusionV3(
            t_low_dim=input_list[0],  # 32
            t_high_dim=input_list[1],  # 64
            out_dim=input_list[1]  # 64
        )

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])# 用于编码器的更高层  特征维度 512

        self.dnorm3 = norm_layer(embed_dims[1]) #embed_dims[1]
        self.dnorm4 = norm_layer(embed_dims[0])

        self.block1 = SwinTransformerStage(dim=embed_dims[1], depth=depths[0], num_heads=8)
        self.block2 = SwinTransformerStage(dim=embed_dims[2], depth=depths[1], num_heads=8)
        self.dblock1 = SwinTransformerStage(dim=embed_dims[1], depth=depths[2], num_heads=10)

        # PatchEmbed 模块，用于将特征图转换为补丁嵌入。 img_size=224输入图像的尺寸为 224x224 像素。
        self.patch_embed3 = PatchEmbed( patch_size=2,  in_c=embed_dims[0],embed_dim=embed_dims[1])  # （256，320）
        self.patch_embed4 = PatchEmbed(patch_size=2,  in_c=embed_dims[1],embed_dim=embed_dims[2])  # （320，512）

        # 最终的卷积层，将特征图转换为类别数。 Softmax层，代码中未使用此参数。
        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1) #256/8=32
        self.soft = nn.Softmax(dim =1)#nn.Softmax 是 PyTorch 中的一个激活函数，用于将输入张量的每个元素转换为一个概率值，使得所有概率值的和为 1。


    def _make_encode_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )
    def _make_bottleneck_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def _make_decode_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    # 只用了小波下采样 和CA
    def forward(self, x):
        B = x.shape[0]

        ### Encoder 编码器阶段
        ### Stage 1
        enc1 = self.encode_layer1(x)
        t1 = enc1

        ### Stage 2
        enc2 = self.down_wt1(enc1)
        enc2 = self.encode_layer2(enc2)
        t2 = enc2

        ### Stage 3
        enc3 = self.down_wt2(enc2)
        enc3 = self.encode_layer3(enc3)
        t3 = enc3

        ### Stage 4

        out, H, W = self.patch_embed3(t3)
        out, H, W = self.block1(out, H, W)
        out = self.norm3(out)
        t4 = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        ### Bottleneck瓶颈阶段
        out, H, W = self.patch_embed4(t4)
        out, H, W = self.block2(out, H, W)
        out = self.norm4(out)
        t5 = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ###解码器阶段
        ### dStage 4

        out = F.relu(
            F.interpolate(self.decode_layer4(t5), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = self.fusion3(t3, out)
        B, C, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out, H, W = self.dblock1(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### dStage 3    [320,256]
        dec3 = F.relu(F.interpolate(self.decode3(out), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        dec3 = self.fusion2(t2, dec3)

        ### dStage 2  256,64
        dec2 = F.relu(F.interpolate(self.decode2(dec3), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        dec2 = self.fusion1(t1, dec2)

        ### dStage 1 #64,32
        dec1 = F.relu(F.interpolate(self.decode1(dec2), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        dec1 = torch.add(dec1, t1)
        dec1 = self.decode_layer1(dec1)

        # Final Stage输出:
        out = self.out_layer(dec1)

        return out
