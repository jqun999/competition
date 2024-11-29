import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR


X = np.random.randn(num_samples, input_dim)  # 传感器数据（输入特征）
y = X.sum(axis=1) * 0.5 + 1.0  # 模拟的力值（目标），简单线性关系

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 数据加载器
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 定义BP神经网络
class ForceSensorNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForceSensorNN, self).__init__()
        # 定义神经网络层结构
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x


# 初始化模型
model = ForceSensorNN(input_dim, output_dim)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # 学习率调度

# 训练模型
num_epochs = 500
best_loss = float('inf')
early_stopping_patience = 50  # 早停的耐心次数
patience_counter = 0

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()

    # 学习率调度
    scheduler.step()

    # 计算平均损失
    avg_loss = running_loss / len(train_loader)

    # 验证模型
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(test_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")


# 测试模型
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
    test_loss /= len(test_loader)

print(f'Test Loss: {test_loss:.4f}')
