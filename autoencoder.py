import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
from encoder import Autoencoder
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Nowplaying/Tmall/yoochoose1_64')
parser.add_argument('--seq_len', type=int, default=4)

opt = parser.parse_args()

# data = np.array([1.4, 1.6, 1.8, 1.9, 2.0, 1.6, 0.4, 1.8])
data = pickle.load(open(f'./datasets/{opt.dataset}/train.txt', 'rb'))
category = pickle.load(open(f'./datasets/{opt.dataset}/category.txt', 'rb'))
score_list = []
N = max([len(i) for i in data[0]])
for i in tqdm(data[0]):
    if len(i) > opt.seq_len:
        score = filter(i, category, 10, 3, 10, flag='min-max', re_score=True)
        mean = np.mean(score)
        score = np.concatenate([score, np.array([mean for i in range(N - len(i))])], axis=0)
        score_list.append(score)


# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(data.reshape(-1, 1))
X = np.array(score_list)  # 因为是单个点序列，不使用时间窗口
X_tensor = torch.tensor(X, dtype=torch.float32)

input_dim = X_tensor.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 800
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)  # 输入整个数据集（多个序列）
    loss = criterion(output, X_tensor)  # 计算重构误差
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), f'./dict/autoencoder-{opt.dataset}.model')
score = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 0.8, 0.4, 0.9])
data = np.concatenate([score, np.array([np.mean(score) for i in range(N - len(score))])], axis=0)
# score = np.concatenate([score, np.array([mean for i in range(N - len(i))])], axis=0)
data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
# 模型评估
model.eval()
with torch.no_grad():
    reconstructed = model(data_tensor).numpy()

reconstruction_error = np.abs(score - reconstructed[0][:len(score)])
print(f"Reconstruction Error: {reconstruction_error}")
print(f"Dim: {input_dim}")
