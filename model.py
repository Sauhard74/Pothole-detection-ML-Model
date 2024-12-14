import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# Define PotholeDetector model
class PotholeDetector(nn.Module):
    def __init__(self):
        super(PotholeDetector, self).__init__()
        # Use ResNet18 with proper weights initialization
        weights = ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 2 classes: Plain and Pothole

    def forward(self, x):
        return self.resnet(x)
