import torch
import torch.nn as nn
import numpy as np
import config


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()

#         self.layer_emb = nn.Embedding(config.num_classes, config.num_classes)

        self.layer_1 = nn.Sequential(
            nn.Linear(config.latent_dim + config.num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_out = nn.Sequential(
            nn.Linear(1024, np.prod(config.img_shape)),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, z):
#         print(noise.shape)
#         print(conditions.shape)
#         print(self.layer_emb(conditions).shape)
        out = self.layer_1(z)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_out(out)

        return out


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()

#         self.layer_emb = nn.Embedding(config.num_classes, config.num_classes)

        self.layer_1 = nn.Sequential(
            nn.Linear(np.prod(config.img_shape) + config.num_classes, 512),
#             nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.layer_out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_out(out)

        return out
