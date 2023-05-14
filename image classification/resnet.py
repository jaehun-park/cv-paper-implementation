import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm="bnorm", relu=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding))
        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU())  
        
        self.convblk = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convblk(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, bias=True, norm="bnorm", relu=True, init_block=False):
        super().__init__()

        if init_block:
          init_stride = 2
        else:
          init_stride = stride

        self.resblk = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, 
                      init_stride, padding, bias, norm, relu),
            ConvBlock(out_channels, out_channels, kernel_size, 
                      stride, padding, bias, norm=None, relu=False)
        )

        if init_block:
            self.short_cut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.short_cut = None


    def forward(self, x):
        out = self.resblk(x)
        if self.short_cut is not None:
            out = self.resblk(x)
            x = self.short_cut(x)
        
        out = nn.ReLU()(x + out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, nker=64, nblk=[3,4,6,3]):
        super(ResNet, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=3, bias=True, norm=None, relu=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self.make_resblk_layers(nker, nker, num=nblk[0], init=False)
        self.conv3_x = self.make_resblk_layers(nker, nker*2, num=nblk[1], init=True)
        self.conv4_x = self.make_resblk_layers(nker*2, nker*4, num=nblk[2], init=True)
        self.conv5_x = self.make_resblk_layers(nker*4, nker*8, num=nblk[3], init=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(nker*8, 10)


    def make_resblk_layers(self, in_channels, out_channels, num, init=True):
        layers = [ResBlock(in_channels, out_channels, init_block=init)] if init else []
        layers += [ResBlock(out_channels, out_channels) for _ in range(num-1)]
        result = nn.Sequential(*layers)
        return result

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out