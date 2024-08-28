import os
import cv2  # Import OpenCV for reading images
import torch
import torchvision.transforms as transforms
from timm import create_model
import numpy as np

# Load the model
model_name = "efficientnet_b0"
model = create_model(model_name, pretrained=False, num_classes=5)
weight = torch.load('./weights2/efficientnet_b0_9.pth', map_location=torch.device('cpu'))
model.load_state_dict(weight['model_state_dict'])

model.eval()  # Set the model to evaluation mode

# Label mapping dictionary
label_mapping = {0: 'airplane', 1: 'bridge', 2: 'palace', 3: 'ship', 4: 'stadium'}

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize with ImageNet stats
])

def predict_single(img):
    """Predicts the label of a single image"""
    # Convert BGR (OpenCV format) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the input size expected by the model (256x256 in this case)
    img = cv2.resize(img, (256, 256))
    
    # Apply transformations: convert to tensor and normalize
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(img)
    
    # Get the predicted class (index with the highest score)
    _, predicted_idx = torch.max(output, 1)
    
    return predicted_idx.item()

def predict(img):
    """Performs Test-Time Augmentation (TTA) by predicting using original, horizontally flipped,
    and vertically flipped images, and returns the final prediction by voting."""
    
    # Original image prediction
    original_pred = predict_single(img)
    
    # Horizontal flip prediction
    horizontal_flip_img = cv2.flip(img, 1)
    horizontal_flip_pred = predict_single(horizontal_flip_img)
    
    # Vertical flip prediction
    vertical_flip_img = cv2.flip(img, 0)
    vertical_flip_pred = predict_single(vertical_flip_img)
    
    # Combine predictions and perform voting
    predictions = [original_pred, horizontal_flip_pred, vertical_flip_pred]
    final_pred = np.bincount(predictions).argmax()  # Majority vote
    
    return label_mapping[final_pred]

def main():
    path = './test'  # Path to the test directory
    images = os.listdir(path)  # List all images in the directory
    
    # Loop through each image in the test folder
    for image_file in images:
        image_path = os.path.join(path, image_file)
        img = cv2.imread(image_path)
        
        # Run inference using TTA on the image
        prediction = predict(img)
        
        # Print the result (image name and its predicted label)
        print(f"Image: {image_file} - Predicted Label: {prediction}")

if __name__ == "__main__":
    main()
