from paddle import nn
import paddle

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    bias = paddle.ParamAttr() if bias else False
    modules = [nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, bias_attr=bias)]
    if bn:
        modules.append(nn.BatchNorm2D(out_channels))
    if relu:
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias_attr=False),
        nn.BatchNorm2D(in_channels),
        nn.ReLU(),

        nn.Conv2D(in_channels, out_channels, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias_attr=False),
        nn.ELU(),

        nn.Conv2D(in_channels, out_channels, 1, 1, 0, bias_attr=False),
        nn.ELU(),
    )