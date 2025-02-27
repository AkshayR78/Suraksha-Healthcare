from PIL import Image
import torch
from torchvision import transforms
from model import SkinDiseaseCNN

# Load trained model
model = SkinDiseaseCNN(num_classes=6)  # Adjust based on your dataset
model.load_state_dict(torch.load("skin_disease_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example
image_path = "uploads/captured_image.jpg"
prediction = predict(image_path)
print(f"Predicted Class: {prediction}")
