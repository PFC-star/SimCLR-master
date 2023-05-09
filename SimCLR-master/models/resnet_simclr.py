import math

import torch.nn as nn
import torchvision.models as models
import torch
from exceptions.exceptions import InvalidBackboneError
import torch.nn.functional as F
import numpy as np
class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        # self.backbone = self._get_basemodel(base_model)

        self.backbone1c =  self._get_basemodel(base_model)
        self.backbone3c =  self._get_basemodel(base_model)
        dim_mlp = self.backbone1c.fc.in_features
        # add mlp projection head
        self.backbone1c.conv1=nn.Sequential(
                                        nn.Conv2d(1,3,kernel_size=(1,1),stride=1),
                                        nn.BatchNorm2d(3,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
)
        self.backbone3c.fc = nn.Sequential(
                                         nn.Linear(dim_mlp, dim_mlp),
                                         nn.ReLU(inplace=True),
                                         self.backbone3c.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        b,c,h,w = x.size()
        if(c==1):
            return self.backbone1c(x)
        elif(c==3):
            return self.backbone3c(x)


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )
# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        # elif isinstance(m, nn.Linear):
        #     m.weight.data.normal_(0, 0.02)
        #     m.bias.data.zero_()

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out)

        else:
            if self.bias is not None:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=None)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
class FC(nn.Module):
    def __init__(
            self,
            size=28,
            style_dim=23,
            n_mlp=3,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        layers = [PixelNorm()]

        for i in range(n_mlp):  # n_mlp=8, style_dim=512， lr_mlp=0.01
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
    def forward(self,input):
        output = self.style(input)
        return output

class EncoderSimCLR(nn.Module):

    def __init__(self, base_model=None, out_dim=1024):
        super(EncoderSimCLR, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = 80
        self.n_c = 10
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)

        self.fc = FC(28, self.latent_dim)
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.latent_dim + self.n_c)
        )
        #
        # initialize_weights(self)
        #
        # if self.verbose:
        #     print("Setting up {}...\n".format(self.name))
        #     print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]

        return zn