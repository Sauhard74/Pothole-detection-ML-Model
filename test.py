import torch
from torchvision import transforms
from PIL import Image
from train_pothole_detector import PotholeDetector  # Import our model class

def load_and_prepare_image(image_path):
    # Define the same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict(image_path, model_path='best_pothole_detector.pth'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = PotholeDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare image
    image_tensor = load_and_prepare_image(image_path).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        # Get prediction probability
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = probabilities[predicted.item()].item() * 100
        
    result = "Pothole" if predicted.item() == 1 else "Plain Road"
    return result, confidence

if __name__ == "__main__":
    # Specify your image path directly here
    image_path = "D:/pothole2/My Dataset/test/Plain/4.jpg"  # Replace this with your actual image path
    
    try:
        result, confidence = predict(image_path)
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error: {str(e)}")