import torch
from torchvision import transforms
from PIL import Image
from model import InjuryClassifierCNN

model = InjuryClassifierCNN()
model.load_state_dict(torch.load('injury_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_injury(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()
