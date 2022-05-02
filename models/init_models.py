import numpy as np
import os
import torch
import time
import torch.nn as nn

from models.inceptionresv2 import InceptionResnetV2Triplet
# from models.mobilenetv2 import MobileNetV2Triplet
from models.resnet import (
    Resnet18Triplet,
    Resnet34Triplet,
    Resnet50Triplet,
    Resnet101Triplet,
    Resnet152Triplet
)

def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
        
    print("Using {} model architecture.".format(model_architecture))
    # print(model)

    return model


def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu
