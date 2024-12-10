import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import os
from PIL import Image
from tqdm import tqdm
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
def load_dataset(root_dir):
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    print(f"Loading {root_dir}: Found {len(dataset)} images in {len(dataset.classes)} classes")
    return dataset

# Dataset paths
train_dir = "D:/pothole2/My Dataset/train"
test_dir = "D:/pothole2/My Dataset/test"

# Load datasets
print("\nLoading datasets...")
train_dataset = load_dataset(train_dir)
test_dataset = load_dataset(test_dir)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Create model
class PotholeDetector(nn.Module):
    def __init__(self):
        super(PotholeDetector, self).__init__()
        # Use ResNet18 with proper weights initialization
        weights = ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        # Replace last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 2 classes: Plain and Pothole
    
    def forward(self, x):
        return self.resnet(x)

# Initialize model
model = PotholeDetector().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'accuracy': f'{100 * correct / total:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_acc:.2f}%')
        print(f'Time: {epoch_time:.2f} seconds')
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_pothole_detector.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        print('-' * 60)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Function to predict single image
def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        
    return "Pothole" if predicted.item() == 1 else "Plain"

# Train the model
if __name__ == "__main__":
    print("\nStarting training...")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total test batches: {len(test_loader)}")
    print('-' * 60)
    
    train_model(model, train_loader, criterion, optimizer)
    
    print("\nTraining completed!")
    print("\nEvaluating final model...")
    final_accuracy = evaluate_model(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_pothole_detector.pth')
    print("\nFinal model saved successfully!")