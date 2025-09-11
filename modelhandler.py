import os
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from typing import Dict, List, Optional


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
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.pool0 = nn.Identity()
        self.features = base.features

        # Dropout opsional sebelum classifier
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base.classifier.in_features, 1)  # Binary output

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

class DenseNet_SE_Early(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.pool0 = nn.Identity()

        self.stem = nn.Sequential(
            base.features.conv0,
            SEBlock(64),  # Early SE block
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
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
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
        self.se_mid = SEBlock(256)     # SE setelah block2 (channel dari transition2)

        self.block3 = base.features.denseblock3
        self.se_late = SEBlock(1024)   # SE setelah block3 (channel sebelum transition3)

        self.trans3 = base.features.transition3
        self.block4 = nn.Sequential(
            base.features.denseblock4,
            base.features.norm5,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)   # Binary classification output

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.se_mid(x)         # SE Block setelah block2
        x = self.block3(x)
        x = self.se_late(x)        # SE Block setelah block3
        x = self.trans3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

class DenseNet_SE_Late(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
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

        self.se = SEBlock(1024)  # Late SE block after all dense blocks

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)  # Binary classification output

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.se(x)  # Apply SE block after block4 (late attention)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# class ModelHandler:
#     def __init__(self, models_dir="models"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Inisialisasi transform gambar (harus sesuai training)
#         self.transform = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#         ])

#         # Daftar model
#         self.models = {
#             "Plain": (DenseNet(), "model_plain_6.pt"),
#             "Early SE": (DenseNet_SE_Early(), "model_earlySE_10.pt"),
#             "Mid SE": (DenseNet_SE_Mid(), "model_midSE_6.pt"),
#             "Late SE": (DenseNet_SE_Late(), "model_lateSE_6.pt"),
#         }

#         # Load state_dict tiap model
#         for name, (model, filename) in self.models.items():
#             path = os.path.join(models_dir, filename)
#             model.load_state_dict(torch.load(path, map_location=self.device))
#             model.to(self.device)
#             model.eval()
#             self.models[name] = model

#     def predict(self, image_path):
#         image = Image.open(image_path).convert("RGB")
#         tensor = self.transform(image).unsqueeze(0).to(self.device)

#         predictions = []
#         total_real_prob = 0
#         total_fake_prob = 0
#         successful = 0

#         for name, model in self.models.items():
#             try:
#                 with torch.no_grad():
#                     logits = model(tensor)
#                     probs = torch.sigmoid(logits).item()
#                     real_prob = probs * 100
#                     fake_prob = (1 - probs) * 100
#                     pred_class = "Real" if probs >= 0.5 else "Fake"
#                     conf = max(real_prob, fake_prob)

#                 predictions.append({
#                     "model_name": name,
#                     "prediction": pred_class,
#                     "confidence": round(conf, 2),
#                     "real_probability": round(real_prob, 2),
#                     "fake_probability": round(fake_prob, 2),
#                 })

#                 total_real_prob += real_prob
#                 total_fake_prob += fake_prob
#                 successful += 1
#             except Exception as e:
#                 predictions.append({
#                     "model_name": name,
#                     "prediction": "Error",
#                     "error": str(e)
#                 })

#         # Ensemble
#         if successful > 0:
#             avg_real = total_real_prob / successful
#             avg_fake = total_fake_prob / successful
#             ensemble_pred = "Real" if avg_real >= avg_fake else "Fake"
#             ensemble_conf = round(max(avg_real, avg_fake), 2)
#         else:
#             avg_real = avg_fake = ensemble_conf = 0
#             ensemble_pred = "Error"

#         return {
#             "ensemble": {
#                 "prediction": ensemble_pred,
#                 "confidence": ensemble_conf,
#                 "real_probability": round(avg_real, 2),
#                 "fake_probability": round(avg_fake, 2),
#             },
#             "individual_predictions": predictions,
#             "total_models": len(self.models),
#             "successful_predictions": successful
#         }

import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

# Import semua class model
from modelhandler import DenseNet, DenseNet_SE_Early, DenseNet_SE_Mid, DenseNet_SE_Late

class ModelHandlerHF:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Link HTTPS langsung
        self.model_urls = {
            "Plain": "https://huggingface.co/nadeeeyz/Plain/resolve/main/model_plain_6.pt",
            "Early SE": "https://huggingface.co/nadeeeyz/EarlySE/resolve/main/model_earlySE_10.pt",
            "Mid SE": "https://huggingface.co/nadeeeyz/Mid/resolve/main/model_midSE_6.pt",
            "Late SE": "https://huggingface.co/nadeeeyz/Late/resolve/main/model_lateSE_6.pt",
        }

        self.models_cache = {}

    def load_model(self, name):
        if name in self.models_cache:
            return self.models_cache[name]

        url = self.model_urls[name]
        r = requests.get(url)
        r.raise_for_status()
        buffer = BytesIO(r.content)

        # Buat instance model sesuai tipe
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

        # Load state_dict
        state_dict = torch.load(buffer, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.models_cache[name] = model
        return model

    def predict(self, image_file):
        image = Image.open(image_file).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        predictions = []
        total_real_prob = 0
        total_fake_prob = 0
        successful = 0

        for name in self.model_urls:
            try:
                model = self.load_model(name)
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

            except Exception as e:
                predictions.append({
                    "model_name": name,
                    "prediction": "Error",
                    "real_probability": 0.0,
                    "fake_probability": 0.0,
                    "confidence": 0.0,
                    "error": str(e)
                })

        # Ensemble
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

# class ModelHandlerHF:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.transform = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         # Hanya simpan URL Hugging Face model
#         self.model_urls = {
#             "Plain": "https://huggingface.co/nadeeeyz/Plain/resolve/main/model_Plain_6.pt",
#             "Early SE": "https://huggingface.co/nadeeeyz/EarlySE/resolve/main/model_earlySE_10.pt",
#             "Mid SE": "https://huggingface.co/nadeeeyz/Mid/resolve/main/model_MidSE_6.pt",
#             "Late SE": "https://huggingface.co/nadeeeyz/Late/resolve/main/model_LateSE_6.pt",
#         }
#         # Cache model di memori untuk reuse
#         self.models_cache = {}

#     def load_model(self, name):
#         if name in self.models_cache:
#             return self.models_cache[name]

#         # Download file dari Hugging Face
#         url = self.model_urls[name]
#         r = requests.get(url)
#         buffer = BytesIO(r.content)
#         model = torch.load(buffer, map_location=self.device)
#         model.to(self.device)
#         model.eval()
#         self.models_cache[name] = model
#         return model

#     def predict(self, image_file):
#         image = Image.open(image_file).convert("RGB")
#         tensor = self.transform(image).unsqueeze(0).to(self.device)

#         predictions = []
#         total_real_prob = 0
#         total_fake_prob = 0
#         successful = 0

#         for name in self.model_urls:
#             try:
#                 model = self.load_model(name)
#                 with torch.no_grad():
#                     logits = model(tensor)
#                     probs = torch.sigmoid(logits).item()
#                     real_prob = probs * 100
#                     fake_prob = (1 - probs) * 100
#                     pred_class = "Real" if probs >= 0.5 else "Fake"
#                     conf = max(real_prob, fake_prob)

#                 predictions.append({
#                     "model_name": name,
#                     "prediction": pred_class,
#                     "confidence": round(conf, 2),
#                     "real_probability": round(real_prob, 2),
#                     "fake_probability": round(fake_prob, 2),
#                 })
#                 total_real_prob += real_prob
#                 total_fake_prob += fake_prob
#                 successful += 1
#             except Exception as e:
#                 predictions.append({
#                     "model_name": name,
#                     "prediction": "Error",
#                     "error": str(e)
#                 })

#         # Ensemble
#         if successful > 0:
#             avg_real = total_real_prob / successful
#             avg_fake = total_fake_prob / successful
#             ensemble_pred = "Real" if avg_real >= avg_fake else "Fake"
#             ensemble_conf = round(max(avg_real, avg_fake), 2)
#         else:
#             avg_real = avg_fake = ensemble_conf = 0
#             ensemble_pred = "Error"

#         return {
#             "ensemble": {
#                 "prediction": ensemble_pred,
#                 "confidence": ensemble_conf,
#                 "real_probability": round(avg_real, 2),
#                 "fake_probability": round(avg_fake, 2),
#             },
#             "individual_predictions": predictions,
#             "total_models": len(self.model_urls),
#             "successful_predictions": successful
#         }
