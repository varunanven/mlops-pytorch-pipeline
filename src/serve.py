import io
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from src.model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

app = FastAPI()
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2023, 0.1994, 0.2010]
    )
])

config_path = Path("/app/configs/training_config.yaml")
if not config_path.exists():
    config_path = Path("configs/training_config.yaml")
config = load_config(str(config_path))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["data"]["num_classes"],
    ).to(device)

MODEL_PATH = "/app/checkpoints/cifar10_model.pt"
try:
    model_dict = torch.load(MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded configuration error")
    return {"status": "healthy"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    
    try:
        image_bytes = await image.read() 
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = CLASSES[predicted_idx.item()]
            
        return {
            "filename": image.filename,
            "prediction": predicted_class,
            "class_index": predicted_idx.item()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
