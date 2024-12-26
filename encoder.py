import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # 编码器：将输入压缩为低维表示
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),  # 输入维度是L，隐藏层4
            nn.ReLU(),
            nn.Linear(4, 2),  # 压缩到维度2
        )
        
        # 解码器：将低维表示解码回原始数据
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),  # 解压缩到维度4
            nn.ReLU(),
            nn.Linear(4, input_dim),  # 输出维度为L
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded