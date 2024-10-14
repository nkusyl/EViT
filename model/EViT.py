import torch
import torch.nn as nn
from functools import partial
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class BFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.Batch_Norm_1 = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.dwconv_2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.Batch_Norm_2 = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.act_layer = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.drop(x)

        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x_1 = self.dwconv_1(x)
        x_1 = self.act_layer(x_1)
        x_1 = self.Batch_Norm_1(x_1)

        x_2 = self.dwconv_2(x_1)
        x_2 = self.act_layer(x_2)
        x_2 = self.Batch_Norm_2(x_2)
        out = x_1 + x_2

        out = out.flatten(2).permute(0, 2, 1)
        out = self.fc2(out)
        out = self.drop(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        if self.sr_ratio > 1:
            self.q_1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.q_2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.proj_1 = nn.Linear(dim, dim)
            self.proj_2 = nn.Linear(dim, dim)
            self.act = nn.GELU()
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.sr_ratio > 1:
            self.sr_1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True)
            self.norm_1 = nn.LayerNorm(dim)
            self.sr_2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True)
            self.norm_2 = nn.LayerNorm(dim)

    def forward(self, x, H, W):

        B, N, C = x.shape

        if self.sr_ratio > 1:
            q_1 = self.q_1(x).reshape(B, N, self.num_heads, (C) // self.num_heads).permute(0, 2, 1,
                                                                                           3)
            x_1 = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm_1(self.sr_1(x_1).reshape(B, C, -1).permute(0, 2, 1)))
            k_1 = self.k_1(x_1).reshape(B, -1, self.num_heads, (C) // self.num_heads).permute(0, 2, 1,
                                                                                              3)
            v_1 = self.v_1(x_1).reshape(B, -1, self.num_heads, (C) // self.num_heads).permute(0, 2, 1, 3)

            attn_1 = (q_1 @ k_1.transpose(-2, -1)) * self.scale
            attn_1 = attn_1.softmax(dim=-1)
            attn_1 = self.attn_drop(attn_1)
            x_1 = (attn_1 @ v_1).transpose(1, 2).reshape(B, N, (C))
            x_1 = self.proj_1(x_1)
            x_1 = self.proj_drop(x_1)

            q_2 = self.q_2(x_1).reshape(B, N, self.num_heads, (C) // self.num_heads).permute(0, 2, 1,
                                                                                             3)
            x_2 = x_1.permute(0, 2, 1).reshape(B, (C), H, W)
            x_2 = self.act(self.norm_2(self.sr_2(x_2).reshape(B, (C), -1).permute(0, 2, 1)))
            k_2 = self.k_2(x_2).reshape(B, -1, self.num_heads, (C) // self.num_heads).permute(0, 2, 1,
                                                                                              3)
            v_2 = self.v_2(x_2).reshape(B, -1, self.num_heads, (C) // self.num_heads).permute(0, 2, 1,
                                                                                              3)

            attn_2 = (q_2 @ k_2.transpose(-2, -1)) * self.scale
            attn_2 = attn_2.softmax(dim=-1)
            attn_2 = self.attn_drop(attn_2)
            x_2 = (attn_2 @ v_2).transpose(1, 2).reshape(B, N, (C))
            x_2 = x_1 + x_2

            x = self.proj_2(x_2)
            x = self.proj_drop(x)

        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = BFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class EViT(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dims=[46, 92, 184, 368], stem_channel=16, fc_dim=1280,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None, depths=[2, 2, 10, 2], qk_ratio=1,
                 sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        self.head_norm = norm_layer(embed_dims[3])
        self.head_avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W)

        x = self.head_norm(x)
        x = self.head_avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_backbone(backbone_name, num_classes):
    if backbone_name == 'EViT_Tiny':
        backbone = EViT(img_size=224, num_classes=num_classes, embed_dims=[56, 112, 224, 448], stem_channel=28,
                        fc_dim=1280, num_heads=[1, 2, 4, 8], mlp_ratios=[3, 3, 3, 3], qkv_bias=True, qk_scale=None,
                        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, dp=0.0,
                        norm_layer=None, depths=[2, 4, 8, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1])
    elif backbone_name == 'EViT_Small':
        backbone = EViT(img_size=224, num_classes=num_classes, embed_dims=[64, 128, 256, 512], stem_channel=32,
                        fc_dim=1280, num_heads=[1, 2, 4, 8], mlp_ratios=[3, 3, 3, 3], qkv_bias=True, qk_scale=None,
                        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, dp=0.0,
                        norm_layer=None, depths=[3, 5, 15, 3], qk_ratio=1, sr_ratios=[8, 4, 2, 1])
    elif backbone_name == 'EViT_Base':
        backbone = EViT(img_size=224, num_classes=num_classes, embed_dims=[64, 128, 256, 512], stem_channel=32,
                       fc_dim=1280, num_heads=[2, 4, 8, 16], mlp_ratios=[3.5, 3.5, 3.5, 3.5], qkv_bias=True, qk_scale=None,
                       drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, dp=0.0,
                       norm_layer=None, depths=[4, 8, 27, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1])
    elif backbone_name == 'EViT_Large':
        backbone = EViT(img_size=256, num_classes=num_classes, embed_dims=[72, 144, 288, 576], stem_channel=36,
                        fc_dim=1280, num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, dp=0.0,
                        norm_layer=None, depths=[4, 8, 30, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1])
    else:
        raise NotImplementedError
    return backbone


