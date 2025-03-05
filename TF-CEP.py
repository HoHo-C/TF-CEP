import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


class PowerDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data_from_file(file_path, batch_size=32):
    dataset = PowerDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class Generator(nn.Module):
    """ 生成器 """
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """ 判别器 """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(TemporalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(input_dim * 32, 128)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FrequencyFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x_freq = torch.fft.fft(x, dim=-1).abs()  # 进行傅里叶变换并取幅度谱
        x = F.relu(self.fc1(x_freq))
        return self.fc2(x)



class ContrastiveLearning(nn.Module):

    def __init__(self, feature_dim, temperature=0.1):
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, z1, z2):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        sim = F.cosine_similarity(z1, z2, dim=-1) / self.temperature
        loss = -torch.mean(torch.log(torch.exp(sim) / torch.sum(torch.exp(sim))))
        return loss



class CarbonEmissionPredictor(nn.Module):

    def __init__(self, feature_dim):
        super(CarbonEmissionPredictor, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features):
        attn_out, _ = self.attn(features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
        return self.fc(attn_out.mean(dim=0))


def train_model(dataloader, temporal_extractor, frequency_extractor, contrastive_model, predictor, epochs=10):
    optimizer = optim.Adam(
        list(temporal_extractor.parameters()) +
        list(frequency_extractor.parameters()) +
        list(predictor.parameters()), lr=0.001
    )

    for epoch in range(epochs):
        total_loss = 0
        for batch, labels in dataloader:
            temp_feat = temporal_extractor(batch)
            freq_feat = frequency_extractor(batch)

            # 对比学习损失
            loss_contrast = contrastive_model(temp_feat, freq_feat)

            # 预测任务
            fused_features = temp_feat + freq_feat
            preds = predictor(fused_features)
            loss_pred = F.mse_loss(preds, labels)

            # 总损失
            loss = loss_contrast + loss_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


def main():

    file_path = "data/carbon_emission_data.csv"
    dataloader = load_data_from_file(file_path)

    input_dim = next(iter(dataloader))[0].shape[1]
    temporal_extractor = TemporalFeatureExtractor(input_dim)
    frequency_extractor = FrequencyFeatureExtractor(input_dim)
    contrastive_model = ContrastiveLearning(feature_dim=128)
    predictor = CarbonEmissionPredictor(feature_dim=128)

    # 训练模型
    train_model(dataloader, temporal_extractor, frequency_extractor, contrastive_model, predictor, epochs=10)


if __name__ == "__main__":
    main()

