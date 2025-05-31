#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# In[4]:


class MalariaResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(MalariaResNet50, self).__init__()
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace final fully connected layer for binary classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def predict(self, image_path, device='cpu', show_image=False):
        """
        Predict class of a single image.

        Args:
            image_path (str): Path to input image
            device (torch.device): 'cuda' or 'cpu'
            show_image (bool): Whether to display the image

        Returns:
            pred_label (str): "Infected" or "Uninfected"
            confidence (float): Confidence score (softmax output)
        """
        from torchvision import transforms
        from PIL import Image
        import matplotlib.pyplot as plt

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Inference
        self.eval()
        with torch.no_grad():
            output = self(img_tensor)
            probs = F.softmax(output, dim=1)
            _, preds = torch.max(output, 1)

        pred_idx = preds.item()
        confidence = probs[0][pred_idx].item()

        classes = ['Uninfected', 'Infected']
        pred_label = classes[pred_idx]

        if show_image:
            plt.imshow(img)
            plt.title(f"Predicted: {pred_label} ({confidence:.2%})")
            plt.axis("off")
            plt.show()

        return pred_label, confidence

    def save(self, path):
        """Save model state dict"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model state dict from file"""
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")


# In[ ]:




