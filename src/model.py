# src/model.py

import torch
import torch.nn as nn
from torchvision import models


# =========================
# Constants
# =========================

NUM_CLASSES = 43


# =========================
# Model factory function
# =========================

def get_model(pretrained: bool = False) -> nn.Module:
    """
    Creates MobileNetV2 model adapted for GTSRB classification.

    Args:
        pretrained (bool): whether to use pretrained ImageNet weights

    Returns:
        nn.Module: MobileNetV2 model
    """

    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    # Load base model
    model = models.mobilenet_v2(weights=weights)

    # Replace classifier layer
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model


# =========================
# Utility: move to device
# =========================

def load_model_to_device(device: torch.device,
                         pretrained: bool = False) -> nn.Module:
    """
    Creates model and moves it to device.

    Args:
        device: cpu or cuda
        pretrained: use pretrained weights

    Returns:
        model on device
    """

    model = get_model(pretrained=pretrained)

    model = model.to(device)

    return model


# =========================
# Debug / standalone test
# =========================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_to_device(device)

    print("\nModel created successfully\n")

    print(model)

    print("\nDevice:", device)

    # test forward pass
    x = torch.randn(1, 3, 64, 64).to(device)

    y = model(x)

    print("\nOutput shape:", y.shape)