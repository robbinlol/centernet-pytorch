# 添加注意力
import torch
from torch import nn

class CAMoudle(nn.Module):
    def __init__(self, c, h, w, reduction=16):
        super(CAMoudle, self).__init__()
        # b * c * h * w
        self.x_avg = nn.AdaptiveAvgPool2d((h, 1))
        self.y_avg = nn.AdaptiveAvgPool2d((1, w))
        self.conv_1x1 = nn.Conv2d(c, c // reduction, kernel_size=1, stride=1, bias=False)
        self.conv_h = nn.Conv2d(c // reduction, c, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(c // reduction, c, kernel_size=1, stride=1, bias=False)
        self.sh = nn.Sigmoid()


    def forward(self, x):
        _, C, H, W = x.shape# 1, 128, 64, 64
        residual= x# b * c * h * w

        x_cord = self.x_avg(x)# b * c * h * 1
        x_cord = x.permutate(0, 1, 3, 2)
        y_cord = self.y_avg(x)# b * c * 1 * w
        x = torch.cat((x_cord, y_cord), dim=3)# b * c * 1 * (h + w)
        x = self.conv_1x1(x) # b * c//16 * 1 * (h + w)
        atten_h = self.conv_h(x)# b * c * 1 * (h + w)
        atten_w = self.conv_w(x) # b * c *1 * (h + w)


if __name__ == '__main__':
    # data = torch.rand(1, 128, 64, 64)
    # xx,yy = data.split((32, 32), dim=3)
    # print(xx.shape, yy.shape)
    # print(data.shape)
    from nets import ResNet, Bottleneck
    image = torch.rand(1, 3, 512, 512)
    net = ResNet(Bottleneck, [3, 4, 6, 3])
    print(net(image).shape)
