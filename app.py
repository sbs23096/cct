import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    label = "cat" if predicted.item() == 0 else "dog"
    return {"prediction": label}
