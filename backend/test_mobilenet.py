import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define categories (Ensure they match your dataset)
CATEGORIES = ['Cuts', 'Bruises', 'Burns', 'Abrasions', 
              'Laserations', 'Normal', 'Diabetic Wounds', 'Surgical Wounds', 
              'Pressure Wound', 'Venous Wounds']

# Load trained model
model = models.mobilenet_v2(pretrained=False)

# Modify classifier for 10 classes
model.classifier[1] = torch.nn.Linear(1280, len(CATEGORIES))

# Load trained weights
model.load_state_dict(torch.load("mobilenetv2_injury_model_10class.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformations (Same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to predict injury type
def predict_injury(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return CATEGORIES[predicted.item()]

# Test with a sample image
image_path = "dataset/test/Bruises/bruises (3).jpg"  # Replace with actual test image path
predicted_class = predict_injury(image_path)
print(f"Predicted Injury: {predicted_class}")
