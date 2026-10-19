from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import io

app = FastAPI()

with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.convnext_base(weights=None)
model.classifier[2] = nn.Linear(
    model.classifier[2].in_features, 101
)
model.load_state_dict(torch.load(f='model101.pth', map_location=device))
model.to(device)

model.eval()
def predict_label(image):
    with torch.inference_mode():
        img = img_transforms(image).unsqueeze(0).to(device)
        logits = model(img)
        class_name = classes[logits.argmax(dim=1).item()]
        conf = logits.softmax(dim=1).max()

    if conf < 0.50:
        return 'Unable to make a confident prediction.'
    return f'food name = {class_name} ,confidence = {(conf*100):.2f}%.'


@app.get("/")
def home():
    return 'Welcome to food classifier'


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return predict_label(image)