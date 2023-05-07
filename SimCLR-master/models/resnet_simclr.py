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
class EncoderSimCLR(nn.Module):
    def __init__(self, base_model=None, out_dim=128):
        super(EncoderSimCLR, self).__init__()

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = out_dim

        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)


        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            torch.nn.Linear(self.iels,  self.latent_dim)
            # 删掉该层
            # nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)




    def forward(self, in_feat):
        z_img = self.model(in_feat)


        return z_img
