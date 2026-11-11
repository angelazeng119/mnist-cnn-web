import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import (datasets, transforms)
from model import SimpleCNN
# ========== 0. 配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# ========== 1. 数据预处理与加载 ==========
# MNIST 是单通道 28x28 的灰度图，像素值 0~255
# ToTensor() 会把它转成 [0,1] 的浮点张量，并变成 (C, H, W) 格式
# Normalize 用 MNIST 官方的均值和标准差做标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ========== 2. 定义 WelcomeScreen 模型 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积块：1 通道 -> 32 通道，3x3 卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 28x28 -> 14x14
        )
        # 第二个卷积块：32 通道 -> 64 通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 14x14 -> 7x7
        )
        # 全连接分类头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),                 # 随机丢弃一半神经元，缓解过拟合
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)               # 10 个数字类别
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x  # 输出 logits，未过 softmax

model = SimpleCNN().to(device)

# ========== 3. 损失函数与优化器 ==========
criterion = nn.CrossEntropyLoss()  # 内部自带 Softmax，所以模型直接输出 logits
optimizer = optim.Adam(model.parameters(), lr=LR)

# ========== 4. 训练函数 ==========
def train_one_epoch(epoch):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # 清空上一步梯度
        outputs = model(images)          # 前向传播
        loss = criterion(outputs, labels)
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_dataset)
    acc = correct / len(train_dataset)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

# ========== 5. 测试函数 ==========
@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(test_dataset)
    print(f"Test Acc: {acc:.4f}")
    return acc

# ========== 6. 主循环 ==========
if __name__ == "__main__":
    print(f"Using device: {device}")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mnist_cnn_best.pth")

    print(f"Best Test Acc: {best_acc:.4f}")