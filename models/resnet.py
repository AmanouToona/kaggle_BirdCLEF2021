import numpy as np

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchlibrosa.stft import LogmelFilterBank, Spectrogram
# from torchlibrosa.augmentation import SpecAugmentation


class Resnet1(nn.Module):
    def __init__(self, base_name: str = 'resnext50_32x4d', pretrained: bool =False, classes: int = 397):
        super().__init__()
        self.base_name = base_name
        self.pretrained = pretrained

        base_model = timm.create_model(base_name, pretrained=pretrained)
        fc_in_features = base_model.fc.in_features

        # # remove global pooling and head classifier
        base_model.reset_classifier(0)  # base_model.fc = nn.Identity() に等しい
        self.base_model = base_model

        self.head_fc = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_in_features, classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.head_fc(x)
        return x


class Resnet2(nn.Module):
    def __init__(self, base_name: str = 'resnext50_32x4d', pretrained: bool =False, classes: int = 397):
        super().__init__()
        self.base_name = base_name
        self.pretrained = pretrained

        base_model = timm.create_model(base_name, pretrained=pretrained)
        fc_in_features = base_model.fc.in_features

        # # remove global pooling and head classifier
        base_model.reset_classifier(0)  # base_model.fc = nn.Identity() に等しい
        self.base_model = base_model

        self.head_fc = nn.Linear(fc_in_features, classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.head_fc(x)
        return x