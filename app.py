import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model import PotholeDetector  # Import the model class
from torchvision import transforms

# Initialize FastAPI
app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_pothole_detector.pth"
model = PotholeDetector()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()  # Set to evaluation mode

# Define image transformation (same as in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to process the image
def preprocess_image(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Health check endpoint
@app.get("/")
def health_check():
    return {"message": "API is running!"}

# Endpoint to predict pothole or plain road
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(pil_image)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted.item()].item() * 100

        # Interpret the prediction
        result = "Pothole" if predicted.item() == 1 else "Plain Road"
        return {"result": result, "confidence": f"{confidence:.2f}%"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
