import torch
from PIL import Image
from torchvision import transforms
from mnist_cnn import SimpleCNN, device   # 复用你训练脚本里的模型定义


# 预处理必须和训练时完全一致
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 强制转单通道
    transforms.Resize((28, 28)),                  # 缩放到 28x28
    transforms.ToTensor(),                        # -> [0,1]，形状 (1,28,28)
    transforms.Normalize((0.1307,), (0.3081,))    # MNIST 标准化
])

def predict(img_path, model_path="mnist_cnn_best.pth"):
    img = Image.open(img_path)

    # 1) 预处理并加上 batch 维度 -> (1, 1, 28, 28)
    tensor = transform(img).unsqueeze(0).to(device)

    # 2) 加载训练好的权重
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) 推理
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    print(f"图片: {img_path}")
    print(f"预测数字: {pred}  (置信度 {probs[pred]*100:.2f}%)")
    print("各类概率:", {str(i): f"{probs[i]*100:.1f}%" for i in range(10)})
    return pred

if __name__ == "__main__":
    import sys
    predict(sys.argv[1] if len(sys.argv) > 1 else "my_digit.png")