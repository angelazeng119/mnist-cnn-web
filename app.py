import io
import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

app = Flask(__name__)

DEVICE = torch.device("cpu")  # Render 免费版无 GPU，直接锁 CPU
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn_best.pth")

# 预处理必须和训练时完全一致
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 启动时加载一次模型，避免每次请求都重载
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(request.url)

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("L")
    except Exception:
        return render_template("result.html", error="无法读取图片，请上传 png/jpg 格式的图片"), 400

    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred = probs.argmax().item()

    top = {str(i): round(probs[i].item() * 100, 2) for i in range(10)}
    return render_template("result.html", pred=pred, probs=top,
                           confidence=round(probs[pred].item() * 100, 2))


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)