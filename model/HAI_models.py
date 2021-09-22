import torch
import torch.nn as nn

from HolisticAttention import HA
from vgg import B2_VGG

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class HAIM(nn.Module):
    def __init__(self, in_channel):
        super(HAIM, self).__init__()
        self.relu = nn.ReLU(True)
        self.rgb_branch1 = BasicConv2d(in_channel, in_channel/4, 3, padding=1, dilation=1)
        self.rgb_branch2 = BasicConv2d(in_channel, in_channel/4, 3, padding=3, dilation=3)
        self.rgb_branch3 = BasicConv2d(in_channel, in_channel/4, 3, padding=5, dilation=5)
        self.rgb_branch4 = BasicConv2d(in_channel, in_channel/4, 3, padding=7, dilation=7)

        self.d_branch1 = BasicConv2d(in_channel, in_channel/4, 3, padding=1, dilation=1)
        self.d_branch2 = BasicConv2d(in_channel, in_channel/4, 3, padding=3, dilation=3)
        self.d_branch3 = BasicConv2d(in_channel, in_channel/4, 3, padding=5, dilation=5)
        self.d_branch4 = BasicConv2d(in_channel, in_channel/4, 3, padding=7, dilation=7)

        self.rgb_branch1_sa = SpatialAttention()
        self.rgb_branch2_sa = SpatialAttention()
        self.rgb_branch3_sa = SpatialAttention()
        self.rgb_branch4_sa = SpatialAttention()

        self.rgb_branch1_ca = ChannelAttention(in_channel / 4)
        self.rgb_branch2_ca = ChannelAttention(in_channel / 4)
        self.rgb_branch3_ca = ChannelAttention(in_channel / 4)
        self.rgb_branch4_ca = ChannelAttention(in_channel / 4)

        self.r_branch1_sa = SpatialAttention()
        self.r_branch2_sa = SpatialAttention()
        self.r_branch3_sa = SpatialAttention()
        self.r_branch4_sa = SpatialAttention()

        self.r_branch1_ca = ChannelAttention(in_channel / 4)
        self.r_branch2_ca = ChannelAttention(in_channel / 4)
        self.r_branch3_ca = ChannelAttention(in_channel / 4)
        self.r_branch4_ca = ChannelAttention(in_channel / 4)

        self.ca = ChannelAttention(in_channel)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(std=0.01)
        #         m.bias.data.fill_(0)

    def forward(self, x_rgb, x_d):
        x1_rgb = self.rgb_branch1(x_rgb)
        x2_rgb = self.rgb_branch2(x_rgb)
        x3_rgb = self.rgb_branch3(x_rgb)
        x4_rgb = self.rgb_branch4(x_rgb)

        x1_d = self.d_branch1(x_d)
        x2_d = self.d_branch2(x_d)
        x3_d = self.d_branch3(x_d)
        x4_d = self.d_branch4(x_d)

        x1_rgb_ca = x1_rgb.mul(self.rgb_branch1_ca(x1_rgb))
        x1_d_sa = x1_d.mul(self.rgb_branch1_sa(x1_rgb_ca))
        x1_d = x1_d + x1_d_sa
        x1_d_ca = x1_d.mul(self.r_branch1_ca(x1_d))
        x1_rgb_sa = x1_rgb.mul(self.r_branch1_sa(x1_d_ca))
        x2_rgb = x2_rgb + x1_rgb_sa

        x2_rgb_ca = x2_rgb.mul(self.rgb_branch2_ca(x2_rgb))
        x2_d_sa = x2_d.mul(self.rgb_branch2_sa(x2_rgb_ca))
        x2_d = x2_d + x2_d_sa
        x2_d_ca = x2_d.mul(self.r_branch2_ca(x2_d))
        x2_rgb_sa = x2_rgb.mul(self.r_branch2_sa(x2_d_ca))
        x3_rgb = x3_rgb + x2_rgb_sa

        x3_rgb_ca = x3_rgb.mul(self.rgb_branch3_ca(x3_rgb))
        x3_d_sa = x3_d.mul(self.rgb_branch3_sa(x3_rgb_ca))
        x3_d = x3_d + x3_d_sa
        x3_d_ca = x3_d.mul(self.r_branch3_ca(x3_d))
        x3_rgb_sa = x3_rgb.mul(self.r_branch3_sa(x3_d_ca))
        x4_rgb = x4_rgb + x3_rgb_sa

        x4_rgb_ca = x4_rgb.mul(self.rgb_branch4_ca(x4_rgb))
        x4_d_sa = x4_d.mul(self.rgb_branch4_sa(x4_rgb_ca))
        x4_d = x4_d + x4_d_sa
        x4_d_ca = x4_d.mul(self.r_branch4_ca(x4_d))
        x4_rgb_sa = x4_rgb.mul(self.r_branch4_sa(x4_d_ca))


        y = torch.cat((x1_rgb_sa, x2_rgb_sa, x3_rgb_sa, x4_rgb_sa), 1)
        y_ca = y.mul(self.ca(y))

        z = y_ca + x_rgb
        # then try z = y_ca + x_rgb + x_d, choose the better performance

        return z



class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.decoder5 = nn.Sequential(
            BasicConv2d(channel, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 128, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            # BasicConv2d(256, 128, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BasicConv2d(128, 64, 3, padding=1),
            BasicConv2d(64, 32, 3, padding=1),
            # BasicConv2d(256, 128, 3, padding=1),
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)


    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        x5_up = self.decoder5(x5)
        # print('x5_up size {} '.format(x5_up.shape))
        s5 = self.S5(x5_up)

        x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
        # print('x4_up size {} '.format(x4_up.shape))
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        # print('x3_up size {} '.format(x3_up.shape))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        # print('x2_up size {} '.format(x2_up.shape))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        # print('x1_up size {} '.format(x1_up.shape))
        s1 = self.S1(x1_up)
        # print('s1 size {} '.format(s1.shape))

        return s1, s2, s3, s4, s5


class HAIMNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(HAIMNet_VGG, self).__init__()
        #Backbone model

        self.vgg = B2_VGG('rgb')
        self.vgg_dep = B2_VGG('dep')

        self.haim_5 = HAIM(512)
        self.haim_4 = HAIM(512)
        self.haim_3 = HAIM(256)
        self.haim_2 = HAIM(128)
        self.haim_1 = HAIM(64)

        # self.agg2_rgbd = aggregation(channel)
        self.decoder_rgbd = decoder(512)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        self.HA_img = HA()

    def forward(self, x_rgb, x_d):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        x1_d = self.vgg_dep.conv1(x_d)
        x2_d = self.vgg_dep.conv2(x1_d)
        x3_d = self.vgg_dep.conv3(x2_d)
        x4_d = self.vgg_dep.conv4(x3_d)
        x5_d = self.vgg_dep.conv5(x4_d)

        # en means enhance
        x5_en = self.haim_5(x5_rgb, x5_d)
        x4_en = self.haim_4(x4_rgb, x4_d)
        x3_en = self.haim_3(x3_rgb, x3_d)
        x2_en = self.haim_2(x2_rgb, x2_d)
        x1_en = self.haim_1(x1_rgb, x1_d)

        s1, s2, s3, s4 ,s5 = self.decoder_rgbd(x5_en, x4_en, x3_en, x2_en, x1_en)

        s3 = self.upsample2(s3)
        s4 = self.upsample4(s4)
        s5 = self.upsample8(s5)

        return s1, s2, s3, s4, s5, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4), self.sigmoid(s5)
