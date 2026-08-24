import torch.nn as nn
from torchvision import models


def get_model(architecture, num_classes):

    if architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_final_layer = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_final_layer, num_classes)
        return model
    
    elif architecture == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_final_layer = model.fc.in_features
        model.fc = nn.Linear(in_final_layer, num_classes)
        return model

