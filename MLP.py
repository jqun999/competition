import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

lamdaBase=[1531.11	1535.2486	1538.834	1542.8713	1546.562;
    1530.6974	1534.7426	1538.3229	1542.6273	1546.2656;
    1530.4457	1534.7247	1538.359	1542.3279	1545.9535


];

lamdaShift=[
-0.2494	-0.1349	-0.1405	-0.1471	-0.1831;
-0.1199	-0.0492	-0.0984	-0.1466	-0.0443;
0.3942	0.2079	0.2063	0.2018	0.2277


];

# 输入数据是光纤误差，目标数据是校正后的光纤参数
X = np.random.randn(num_samples, input_dim)  # 模拟输入误差
y = X * 0.8 + 0.1  # 模拟校正后的参数，假设有一定的线性关系

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class ErrorCorrectionMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ErrorCorrectionMLP, self).__init__()
        # 定义多层感知机的层结构
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = ErrorCorrectionMLP(input_dim, output_dim)

criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    model.eval()
    predicted = model(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')


