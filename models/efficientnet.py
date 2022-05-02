import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class EfficientNetB0(nn.Module):
    ''' EfficientNet-b0 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_name('efficientnet-b0', num_classes=c_dim)
        
    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out

    

class EfficientNetB1(nn.Module):
    ''' EfficientNet-b1 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_name('efficientnet-b1', num_classes=c_dim)
        
    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out


class EfficientNetB5(nn.Module):
    ''' EfficientNet-b5 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_name('efficientnet-b5', num_classes=c_dim)
        
    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out


class EfficientNetB7(nn.Module):
    ''' EfficientNet-b7 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_name('efficientnet-b7', num_classes=c_dim)
        
    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out