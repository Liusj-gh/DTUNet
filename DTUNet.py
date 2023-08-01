# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)

class up_cov_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(up_cov_block_3d, self).__init__()
        self.upsample = InterpolateUpsampling()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm3d(out_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, encoder_features, x):
        x = self.upsample(encoder_features, x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# def conv_block_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim))
#         #activation, )
#
#
# def conv_trans_block_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
#         nn.BatchNorm3d(out_dim))
#         #activation, )


class conv_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(conv_block_3d, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x


class conv_trans_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(conv_trans_block_3d, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.deconv1 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(out_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.deconv1(x)))
        return x


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d_en(in_dim, out_dim, activation):
    middle_dim = int(out_dim / 2)
    return nn.Sequential(
        conv_block_3d(in_dim, middle_dim, activation),
        conv_block_3d(middle_dim, out_dim, activation))

def conv_block_2_3d_de(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        conv_block_3d(out_dim, out_dim, activation))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FuseBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(FuseBlock, self).__init__()
        self.conv1 = conv_block_3d(inplanes, planes, activation=None)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class Dual_UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters=32):
        super(Dual_UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        #activation = nn.LeakyReLU(0.2, inplace=True)
        activation = None

        # Down sampling
        self.down_1 = conv_block_2_3d_en(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d_en(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d_en(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d_en(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_1 = conv_block_2_3d_de(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_2 = conv_block_2_3d_de(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_3 = conv_block_2_3d_de(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1)

        # Bridge
        self.bridge2 = conv_block_2_3d_en(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_12 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_12 = conv_block_2_3d_de(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_22 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_22 = conv_block_2_3d_de(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_32 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_32 = conv_block_2_3d_de(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out2 = nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        x = self.pool_1(down_1)

        down_2 = self.down_2(x)
        x = self.pool_2(down_2)

        down_3 = self.down_3(x)
        sx = self.pool_3(down_3)

        # Bridge
        x = self.bridge(sx)

        # Up sampling
        # trans_1 = self.trans_1(down_3, bridge)
        x = self.trans_1(x)
        x = torch.cat([x, down_3], dim=1)
        x = self.up_1(x)

        # trans_2 = self.trans_2(down_2, up_1)
        x = self.trans_2(x)
        x = torch.cat([x, down_2], dim=1)
        x = self.up_2(x)

        # trans_3 = self.trans_3(down_1, up_2)
        x = self.trans_3(x)
        x = torch.cat([x, down_1], dim=1)
        x = self.up_3(x)

        # Output
        out = self.out(x)

        # Bridge
        x2 = self.bridge2(sx)

        # Up sampling
        # trans_1 = self.trans_1(down_3, bridge)
        x2 = self.trans_12(x2)
        x2 = torch.cat([x2, down_3], dim=1)
        x2 = self.up_12(x2)

        # trans_2 = self.trans_2(down_2, up_1)
        x2 = self.trans_22(x2)
        x2 = torch.cat([x2, down_2], dim=1)
        x2 = self.up_22(x2)

        # trans_3 = self.trans_3(down_1, up_2)
        x2 = self.trans_32(x2)
        x2 = torch.cat([x2, down_1], dim=1)
        x2 = self.up_32(x2)

        # Output
        out2 = self.out2(x2)

        return out, out2


class DTUNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters=32):
        super(DTUNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        #activation = nn.LeakyReLU(0.2, inplace=True)
        activation = None

        # Down sampling
        self.down_1 = conv_block_2_3d_en(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d_en(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d_en(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d_en(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_1 = conv_block_2_3d_de(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_2 = conv_block_2_3d_de(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_3 = conv_block_2_3d_de(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1)

        # Bridge
        self.bridge2 = conv_block_2_3d_en(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_12 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_12 = conv_block_2_3d_de(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_22 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_22 = conv_block_2_3d_de(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_32 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_32 = conv_block_2_3d_de(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out2 = nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1)

        # share
        self.fb1 = FuseBlock(self.num_filters * 16, self.num_filters * 8)
        self.fb2 = FuseBlock(self.num_filters * 8, self.num_filters * 4)
        self.fb3 = FuseBlock(self.num_filters * 4, self.num_filters * 2)


    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        x = self.pool_1(down_1)

        down_2 = self.down_2(x)
        x = self.pool_2(down_2)

        down_3 = self.down_3(x)
        sx = self.pool_3(down_3)

        # Bridge
        x = self.bridge(sx)
        x2 = self.bridge2(sx)
        fx = torch.cat([x, x2], dim=1)
        fx = self.fb1(fx)
        x = x + fx
        x2 = x2 + fx

        # Up sampling
        # trans_1 = self.trans_1(down_3, bridge)
        x = self.trans_1(x)
        x = torch.cat([x, down_3], dim=1)
        x = self.up_1(x)

        x2 = self.trans_12(x2)
        x2 = torch.cat([x2, down_3], dim=1)
        x2 = self.up_12(x2)

        fx = torch.cat([x, x2], dim=1)
        fx = self.fb2(fx)
        x = x + fx
        x2 = x2 + fx

        # trans_2 = self.trans_2(down_2, up_1)
        x = self.trans_2(x)
        x = torch.cat([x, down_2], dim=1)
        x = self.up_2(x)

        x2 = self.trans_22(x2)
        x2 = torch.cat([x2, down_2], dim=1)
        x2 = self.up_22(x2)

        fx = torch.cat([x, x2], dim=1)
        fx = self.fb3(fx)
        x = x + fx
        x2 = x2 + fx

        # trans_3 = self.trans_3(down_1, up_2)
        x = self.trans_3(x)
        x = torch.cat([x, down_1], dim=1)
        x = self.up_3(x)

        # Output
        out = self.out(x)

        # trans_3 = self.trans_3(down_1, up_2)
        x2 = self.trans_32(x2)
        x2 = torch.cat([x2, down_1], dim=1)
        x2 = self.up_32(x2)

        # Output
        out2 = self.out2(x2)

        return out, out2

