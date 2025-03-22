# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 步骤1: 定义目标函数
def target_function(input_x):
    """目标函数：sin(x) + 0.5x"""
    return np.sin(input_x) + 0.5 * input_x


# 步骤2: 生成数据集（训练集和测试集）
# 生成1000个在[-5, 5]区间均匀分布的数据点
x_values = np.linspace(-5, 5, 1000).reshape(-1, 1)  # 1000 points in range [-5, 5]
y_values = target_function(x_values)

# 将NumPy数组转换为PyTorch张量
x_tensor = torch.tensor(x_values, dtype=torch.float32)
y_tensor = torch.tensor(y_values, dtype=torch.float32)

# 将数据分割为训练集和测试集（80%训练，20%测试）
x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)


# 步骤3: 构建神经网络模型(2-layer ReLU)
class NeuralNet(nn.Module):
    def __init__(self):
        """神经网络结构定义"""
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(1, 64)  # 输入层（1个特征）到第一隐藏层（64个神经元）
        self.layer2 = nn.Linear(64, 64)  # 第一隐藏层到第二隐藏层
        self.output_layer = nn.Linear(64, 1)   # 第二隐藏层到输出层（1个输出）

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # 第一层使用ReLU激活
        x = torch.relu(self.layer2(x))  # 第二层使用ReLU激活
        x = self.output_layer(x)  # 输出层不使用激活函数（回归任务）
        return x


# 初始化模型、损失函数和优化器
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 步骤4: 训练模型
num_epochs = 500
loss_history = []
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train) # 计算损失

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # 每50轮打印损失值
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], 损失: {loss.item():.4f}')


# 步骤5: 评估模型
model.eval()  # 设置模型为评估(evaluation mode)模式（关闭dropout等训练专用层）
with torch.no_grad():# 禁用梯度计算以提升效率
    y_pred = model(x_test)

# Step 6: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_test.numpy(), y_test.numpy(), label='真实值', color='blue', alpha=0.5)
plt.scatter(x_test.numpy(), y_pred.numpy(), label='预测值', color='red', alpha=0.5, marker='x')
plt.title("使用2层ReLU神经网络进行函数近似")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='训练损失', color='green')
plt.title("训练损失曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Print the final loss value
loss = criterion(y_pred, y_test)
#print(f"Final test loss: {loss.item():.4f}")
print(f"测试集最终损失值: {loss.item():.4f}")