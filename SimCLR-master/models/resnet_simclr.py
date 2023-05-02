import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone1c = self.backbone
        self.backbone3c = self.backbone
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
                                         self.backbone.fc)

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

