import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from ResNet import ResNet

import math


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
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

class DM(nn.Module):
    def __init__(self):
        super(DM, self).__init__()
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        self.conv2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            BasicConv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            BasicConv2d(64, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, left, down, num='None'):
        if num != 'one':
            left = self.up2(left)
        
        reb = self.conv1(torch.cat([down, left], dim = 1))
        
        x = -1*(torch.sigmoid(left)) + 1
        x = x.expand(-1, 64, -1, -1).mul(down)
        ramb = self.conv2(x)
        
        x = reb + ramb + left
        
        return x



class PINet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=64):
        super(PINet, self).__init__()

        # ---- ResNet Backbone ----
        self.resnet = ResNet()

        # Receptive Field Block
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)

        
        self.cfm43 = CFM()
        self.cfm32 = CFM()
        self.cfm21 = CFM()
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # ---- reverse attention branch 4 ----
        #self.ra4_conv1 = BasicConv2d(2048, 64, kernel_size=1)
        self.predict4 = nn.Conv2d(64, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        # self.ra4_3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        #self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra3_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16)
        # ---- recurrent refinement branch 3 ----

        self.predict3 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        # ---- reverse attention branch 2 ----
        # self.ra3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        #self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra2_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8)
        # ---- recurrent refinement branch 2 ----

        self.predict2 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra2_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8)
        # ---- recurrent refinement branch 2 ----

        self.predict1 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.ra_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra2_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8)
        # ---- recurrent refinement branch 2 ----

        self.predict = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        # self.HA = HA()
        if self.training:
            self.initialize_weights()
            # self.apply(CRANet.weights_init)
        
        
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)
        
        # bs, 2048, 11, 11
        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        

        ## Origin
        x = self.predict4(x4_rfb)
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')      # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        
        #lateral_map_4 = x.clone()
        # ---- reverse attention branch_3 ----
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        out3h, out3v = self.cfm43(x3_rfb, x4_rfb)


        re3_feat = self.predict3(torch.cat([out3h, crop_3], dim=1))
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 64, -1, -1).mul(out3h)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3 + re3_feat


        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')
        #lateral_map_3 = x.clone()
        # ---- reverse attention branch_2 ----
        # x = self.ra3_2(x)
        # crop_2 = self.crop(x, x2.size())
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        out2h, out2v = self.cfm32(x2_rfb, out3v)


        re2_feat = self.predict2(torch.cat([out2h, crop_2], dim=1))
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 64, -1, -1).mul(out2h)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2 + re2_feat
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')
        #lateral_map_2 = x.clone()
        #lateral_map_2 = self.crop(self.ra2_conv4_up(x), x_size)  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear')

        out1h, out1v = self.cfm21(x1_rfb, out2v)

        re1_feat = self.predict1(torch.cat([out1h, crop_1], dim=1))
        x = -1 * (torch.sigmoid(crop_1)) + 1
        x = x.expand(-1, 64, -1, -1).mul(out1h)
        x = F.relu(self.ra1_conv2(x))
        x = F.relu(self.ra1_conv3(x))
        ra1_feat = self.ra1_conv4(x)
        x = ra1_feat + crop_1 + re1_feat
        lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')
        pred1h = x.clone()
        re_feat = self.predict(torch.cat([out1v, pred1h], dim=1))
        x = -1 * (torch.sigmoid(pred1h)) + 1
        x = x.expand(-1, 64, -1, -1).mul(out1v)
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        ra_feat = self.ra_conv4(x)
        x = ra_feat + pred1h + re_feat
        lateral_map = F.interpolate(x, scale_factor=4, mode='bilinear')
        
        return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map
        

    
    # def crop(self, upsampled, x_size):
    #     c = (upsampled.size()[2] - x_size[2]) // 2
    #     _c = x_size[2] - upsampled.size()[2] + c
    #     assert(c >= 0)
    #     if c == _c == 0:
    #         return upsampled
    #     return upsampled[:, :, c:_c, c:_c]

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        

    # @staticmethod
    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.ConvTranspose2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))



if __name__ == '__main__':
    ras = CRANet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)