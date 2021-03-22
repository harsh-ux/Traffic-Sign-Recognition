import numpy as np

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_filter(kernel_shape):

    x = np.zeros(kernel_shape, dtype='float64')    

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ref:
# https://stackoverflow.com/questions/27948324/implementing-lecun-local-contrast-normalization-with-theano
# FOR LCN


class LCN(nn.Module):
    def __init__(self, channels, radius=9):
        super(LCN, self).__init__()
        self.ch = channels
        self.radius = radius
        self.filter = nn.Parameter(torch.Tensor(get_gaussian_filter(
          (1, channels, radius, radius))).to(device), requires_grad=False)

    def forward(self, image):
        radius = self.radius
        gaussian_filter = self.filter

        filtered_out = F.conv2d(image, gaussian_filter, padding=radius-1)
        mid = int(np.floor(gaussian_filter.shape[2] / 2.))

        # Subtractive Normalization
        centered_image = image - filtered_out[:, :, mid:-mid, mid:-mid]

        # Variance Calc
        sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter,
                                 padding=radius-1)
        s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
        per_img_mean = s_deviation.mean(axis=[2, 3], keepdim=True)

        # Divisive Normalization
        divisor = torch.maximum(per_img_mean, s_deviation)
        divisor_ = torch.maximum(divisor, torch.tensor(1e-4))
        new_image = centered_image / divisor_

        return new_image


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size):
        super(ConvBlock, self).__init__()

        self.lcn_radius = 9

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, filter_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            LCN(out_ch)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class STNBlock(nn.Module):
    def __init__(self, in_ch, md_ch, out_ch, last_conv_out_dim, fc_neurons):
        super(STNBlock, self).__init__()

        self.localization = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_ch, md_ch, 5, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(md_ch, out_ch, 5, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch*last_conv_out_dim*last_conv_out_dim, fc_neurons),
            nn.ReLU(),
            nn.Linear(fc_neurons, 2*3)
        )

        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                       dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)

        x = F.grid_sample(x, grid, align_corners=False)

        return x


class Net(pl.LightningModule):

    def __init__(self, nb_neurons_fc=400, num_classes=43):
        super(Net, self).__init__()
        self.t_loss = nn.CrossEntropyLoss()
        self.v_loss = nn.CrossEntropyLoss()

        self.lcn_preproc = LCN(3)

        self.net = nn.Sequential(
            STNBlock(3, 250, 250, 6, 250),
            ConvBlock(3, 200, 7),
            STNBlock(200, 150, 200, 2, 300),
            ConvBlock(200, 250, 4),
            STNBlock(250, 150, 200, 1, 300),
            ConvBlock(250, 350, 4)
        )

        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(350*6*6, nb_neurons_fc),
            nn.ReLU(),
            nn.Linear(nb_neurons_fc, num_classes)
        )

        # torch.nn.utils.clip_grad_norm_(self.parameters(), 300)

    def forward(self, x):
        x = self.lcn_preproc(x)
        return self.fc_head(self.net(x))

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=1e-2)
        return opt

    def training_step(self, batch, idx):
        x, y = batch
        pred = self(x)
        loss = self.t_loss(pred, y)
        self.log('train_loss', loss)
        return loss
