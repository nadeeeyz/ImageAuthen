import os
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from typing import Dict, List, Optional
import requests
from io import BytesIO
import gc

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class DenseNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=None)  # Don't load ImageNet weights
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.pool0 = nn.Identity()
        self.features = base.features
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base.classifier.in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

class DenseNet_SE_Early(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=None)  # Don't load ImageNet weights
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.pool0 = nn.Identity()

        self.stem = nn.Sequential(
            base.features.conv0,
            SEBlock(64),
            base.features.norm0,
            base.features.relu0,
        )

        self.features = nn.Sequential(
            base.features.denseblock1,
            base.features.transition1,
            base.features.denseblock2,
            base.features.transition2,
            base.features.denseblock3,
            base.features.transition3,
            base.features.denseblock4,
            base.features.norm5,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

class DenseNet_SE_Mid(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=None)  # Don't load ImageNet weights
        base.features.conv0 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        base.features.pool0 = nn.Identity()

        self.stem = nn.Sequential(
            base.features.conv0,
            base.features.norm0,
            base.features.relu0,
        )
        self.block1 = nn.Sequential(
            base.features.denseblock1,
            base.features.transition1,
        )
        self.block2 = nn.Sequential(
            base.features.denseblock2,
            base.features.transition2,
        )
        self.se_mid = SEBlock(256)

        self.block3 = base.features.denseblock3
        self.se_mid1 = SEBlock(1024)

        self.trans3 = base.features.transition3
        self.block4 = nn.Sequential(
            base.features.denseblock4,
            base.features.norm5,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.se_mid(x)
        x = self.block3(x)
        x = self.se_mid1(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

class DenseNet_SE_Late(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=None)  # Don't load ImageNet weights
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.pool0 = nn.Identity()

        self.stem = nn.Sequential(
            base.features.conv0,
            base.features.norm0,
            base.features.relu0,
        )
        self.block1 = nn.Sequential(
            base.features.denseblock1,
            base.features.transition1,
        )
        self.block2 = nn.Sequential(
            base.features.denseblock2,
            base.features.transition2,
        )
        self.block3 = nn.Sequential(
            base.features.denseblock3,
            base.features.transition3,
        )
        self.block4 = nn.Sequential(
            base.features.denseblock4,
            base.features.norm5,
        )

        self.se = SEBlock(1024)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.se(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class OptimizedModelHandlerHF:
    """
    Optimized version that loads models one at a time and clears memory
    """
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU to save memory
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model_urls = {
            "Plain": "https://huggingface.co/nadeeeyz/Plain/resolve/main/model_plain_6.pt",
            "Early SE": "https://huggingface.co/nadeeeyz/EarlySE/resolve/main/model_earlySE_10.pt",
            "Mid SE": "https://huggingface.co/nadeeeyz/Mid/resolve/main/model_midSE_6.pt",
            "Late SE": "https://huggingface.co/nadeeeyz/Late/resolve/main/model_lateSE_6.pt",
        }

        # Don't cache models - load on demand
        self.current_model = None
        self.current_model_name = None

    def _clear_memory(self):
        """Clear current model from memory"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    def _load_model_on_demand(self, name):
        """Load model only when needed and clear previous model"""
        if self.current_model_name == name:
            return self.current_model

        # Clear previous model first
        self._clear_memory()

        url = self.model_urls[name]
        
        # Download with timeout and smaller chunks
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            content = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
            content.seek(0)
        except requests.RequestException as e:
            raise Exception(f"Failed to download model {name}: {str(e)}")

        # Create model instance
        if name == "Plain":
            model = DenseNet()
        elif name == "Early SE":
            model = DenseNet_SE_Early()
        elif name == "Mid SE":
            model = DenseNet_SE_Mid()
        elif name == "Late SE":
            model = DenseNet_SE_Late()
        else:
            raise ValueError(f"Unknown model: {name}")

        try:
            # Load state dict with CPU mapping
            state_dict = torch.load(content, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.current_model = model
            self.current_model_name = name
            
            # Clear the downloaded content
            del content, state_dict
            gc.collect()
            
            return model
        except Exception as e:
            raise Exception(f"Failed to load model {name}: {str(e)}")

    def predict(self, image_file):
        """Predict with memory-efficient approach"""
        try:
            image = Image.open(image_file).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {
                "ensemble": {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "real_probability": 0.0,
                    "fake_probability": 0.0,
                },
                "individual_predictions": [],
                "total_models": len(self.model_urls),
                "successful_predictions": 0,
                "error": f"Image processing error: {str(e)}"
            }

        predictions = []
        total_real_prob = 0
        total_fake_prob = 0
        successful = 0

        # Process models one by one
        for name in self.model_urls:
            try:
                # Load model on demand
                model = self._load_model_on_demand(name)
                
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.sigmoid(logits).item()
                    real_prob = probs * 100
                    fake_prob = (1 - probs) * 100
                    pred_class = "Real" if probs >= 0.5 else "Fake"
                    conf = max(real_prob, fake_prob)

                predictions.append({
                    "model_name": name,
                    "prediction": pred_class,
                    "confidence": round(conf, 2),
                    "real_probability": round(real_prob, 2),
                    "fake_probability": round(fake_prob, 2),
                })

                total_real_prob += real_prob
                total_fake_prob += fake_prob
                successful += 1

                # Clear model after prediction to save memory
                self._clear_memory()

            except Exception as e:
                predictions.append({
                    "model_name": name,
                    "prediction": "Error",
                    "real_probability": 0.0,
                    "fake_probability": 0.0,
                    "confidence": 0.0,
                    "error": str(e)
                })

        # Calculate ensemble
        if successful > 0:
            avg_real = total_real_prob / successful
            avg_fake = total_fake_prob / successful
            ensemble_pred = "Real" if avg_real >= avg_fake else "Fake"
            ensemble_conf = round(max(avg_real, avg_fake), 2)
        else:
            avg_real = avg_fake = ensemble_conf = 0
            ensemble_pred = "Error"

        return {
            "ensemble": {
                "prediction": ensemble_pred,
                "confidence": ensemble_conf,
                "real_probability": round(avg_real, 2),
                "fake_probability": round(avg_fake, 2),
            },
            "individual_predictions": predictions,
            "total_models": len(self.model_urls),
            "successful_predictions": successful
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._clear_memory()


# Alternative: Single Best Model Approach
class SingleModelHandlerHF:
    """
    Use only the best performing model to reduce memory usage
    """
    def __init__(self, model_name="Late SE"):
        self.device = torch.device("cpu")
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        model_urls = {
            "Plain": "https://huggingface.co/nadeeeyz/Plain/resolve/main/model_plain_6.pt",
            "Early SE": "https://huggingface.co/nadeeeyz/EarlySE/resolve/main/model_earlySE_10.pt",
            "Mid SE": "https://huggingface.co/nadeeeyz/Mid/resolve/main/model_midSE_6.pt",
            "Late SE": "https://huggingface.co/nadeeeyz/Late/resolve/main/model_lateSE_6.pt",
        }

        self.model_name = model_name
        self.model_url = model_urls[model_name]
        self.model = None

    def _load_model(self):
        """Load single model"""
        if self.model is not None:
            return self.model

        try:
            response = requests.get(self.model_url, timeout=30)
            response.raise_for_status()
            buffer = BytesIO(response.content)

            # Create model instance
            if self.model_name == "Plain":
                model = DenseNet()
            elif self.model_name == "Early SE":
                model = DenseNet_SE_Early()
            elif self.model_name == "Mid SE":
                model = DenseNet_SE_Mid()
            elif self.model_name == "Late SE":
                model = DenseNet_SE_Late()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            state_dict = torch.load(buffer, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.model = model
            return model
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")

    def predict(self, image_file):
        """Single model prediction"""
        try:
            image = Image.open(image_file).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            model = self._load_model()
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).item()
                real_prob = probs * 100
                fake_prob = (1 - probs) * 100
                pred_class = "Real" if probs >= 0.5 else "Fake"
                conf = max(real_prob, fake_prob)

            return {
                "ensemble": {
                    "prediction": pred_class,
                    "confidence": round(conf, 2),
                    "real_probability": round(real_prob, 2),
                    "fake_probability": round(fake_prob, 2),
                },
                "individual_predictions": [{
                    "model_name": self.model_name,
                    "prediction": pred_class,
                    "confidence": round(conf, 2),
                    "real_probability": round(real_prob, 2),
                    "fake_probability": round(fake_prob, 2),
                }],
                "total_models": 1,
                "successful_predictions": 1
            }
        except Exception as e:
            return {
                "ensemble": {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "real_probability": 0.0,
                    "fake_probability": 0.0,
                },
                "individual_predictions": [],
                "total_models": 1,
                "successful_predictions": 0,
                "error": str(e)
            }
