import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterResponseNormalization(nn.Module):
    
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            num_features: A integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        super(FilterResponseNormalization, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()
    
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        
        if norm_layer is None:
            norm_layer = FilterResponseNormalization

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding,
                               groups=inplanes, bias=False)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class SepConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1,stride=1, padding=0, norm_layer=None):
        super(SepConvBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = FilterResponseNormalization


        self.relu = nn.ReLU(inplace=True)
        self.sepconv = SeparableConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = norm_layer(planes)
        
    def forward(self, inp):
        x = self.relu(inp)
        x = self.sepconv(x)
        
        out = self.bn(x)

        return out

class Block(nn.Module):  # Mid 
    def __init__(self, channels):
        super(Block, self).__init__()
        
        self.sepconv1 = SepConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.sepconv2 = SepConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.sepconv3 = SepConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inp):
        indentity = inp
        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        
        out = x + indentity
        return out 

class DownBlock(nn.Module): # Entry
    def __init__(self, inplanes, outplanes):
        super(DownBlock, self).__init__()
        self.sepconv1 = SepConvBlock(inplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.sepconv2 = SepConvBlock(outplanes, outplanes, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.skip = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=2, padding=0)
        
    def forward(self, inp):
        identity = self.skip(inp)
        
        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        x = self.pool(x)
        
        out = x + identity
        
        return out 

class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, norm_layer=None):
        super(Xception, self).__init__()
        
        if norm_layer is None:
            norm_layer = FilterResponseNormalization


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = DownBlock(64, 128)
        self.block2 = DownBlock(128, 256)
        self.block3 = DownBlock(256, 728)

        # Middle flow
        self.block4  = Block(728)
        self.block5  = Block(728)
        self.block6  = Block(728)
        self.block7  = Block(728)
        self.block8  = Block(728)
        self.block9  = Block(728)
        self.block10 = Block(728)
        self.block11 = Block(728)

        # Exit flow
        self.sepconv12 = SepConvBlock(728, 728, kernel_size=3, stride=1, padding=1)
        self.sepconv13 = SepConvBlock(728, 1024, kernel_size=3, stride=1, padding=1)
        self.skip15 = nn.Conv2d(728, 1024,kernel_size=1, stride=1, padding=0)
        
        
        self.conv16 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1)
        self.bn16 = norm_layer(1536)
        self.relu16 = nn.ReLU(inplace=True)

        self.conv17 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1)
        self.bn17 = norm_layer(2048)
        self.relu17 = nn.ReLU(inplace=True)
        



    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.block1(x)

        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        out = self.sepconv12(x)
        out = self.sepconv13(out)
        skip = self.skip15(x)
        x = out + skip
        
        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu16(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu17(x)


        return x, low_level_feat


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer=None):
        super(_ASPPModule, self).__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, norm_layer=None):
        super(ASPP, self).__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization

        inplanes = 2048

        dilations = [1, 6, 12, 18]


        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             norm_layer(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, num_classes, norm_layer=None):
        super(Decoder, self).__init__()
        
        if norm_layer is None:
            norm_layer = FilterResponseNormalization

        low_level_inplanes = 128

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

class DeepLab(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLab, self).__init__()

        self.n_classes = num_classes
        self.backbone = Xception()
        self.aspp = ASPP()
        self.decoder = Decoder(num_classes)

        
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

model = DeepLab(5)
input = torch.rand(1, 3, 512, 512)
output = model(input)
print(output.size())
