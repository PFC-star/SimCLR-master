import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


# class GaussianBlur(object):
#     """blur a single image on CPU"""
#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         # 为适应 mnist已经修改
#         self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias
#
#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )
#
#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()
#
#     def __call__(self, img):
#         img = self.pil_to_tensor(img).unsqueeze(0)
#
#         sigma = np.random.uniform(0.1, 2.0)
#         x = np.arange(-self.r, self.r + 1)
#         x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#         x = x / x.sum()
#         x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
#
#         self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
#         self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
#
#         with torch.no_grad():
#             img = self.blur(img)
#             img = img.squeeze()
#
#         img = self.tensor_to_pil(img)
#
#         return img
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.k = kernel_size
        self.r = radias

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.blur_h = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=1)
        self.blur_v = nn.Conv2d(1, 1, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=1)

        self.blur_h_3 = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                  stride=1, padding=0, bias=False, groups=3)
        self.blur_v_3 = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                  stride=1, padding=0, bias=False, groups=3)

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            nn.BatchNorm2d(1),
            self.blur_h,
            self.blur_v,
            nn.BatchNorm2d(1)
        )

        self.blur_3 = nn.Sequential(
            nn.ReflectionPad2d(radias),
            nn.BatchNorm2d(3),
            self.blur_h_3,
            self.blur_v_3,
            nn.BatchNorm2d(3)
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img_tensor = self.pil_to_tensor(img).unsqueeze(0)
        c, h, w = img_tensor.size()[1:]

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(c, 1)

        if c == 1:
            self.blur_h.weight.data.copy_(x.view(1, 1, self.k, 1))
            self.blur_v.weight.data.copy_(x.view(1, 1, 1, self.k))

            with torch.no_grad():
                img_tensor = self.blur(img_tensor)
                img_tensor = img_tensor.squeeze()

        elif c == 3:
            self.blur_h_3.weight.data.copy_(x.view(3, 1, self.k, 1))
            self.blur_v_3.weight.data.copy_(x.view(3, 1, 1, self.k))

            with torch.no_grad():
                img_tensor = self.blur_3(img_tensor)

        img = self.tensor_to_pil(img_tensor)

        return img
