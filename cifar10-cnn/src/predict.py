import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CNN

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Load trained model
model = CNN().to(device)
model.load_state_dict(torch.load('model/cnn_cifar10.pth', map_location=device))
model.eval()

# Image preprocessing — must match training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_from_file(image_path):
    """Predict class of a local image file"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    print(f"\n🔍 Image: {image_path}")
    print(f"✅ Prediction: {classes[predicted_idx].upper()}")
    print(f"📊 Confidence: {probabilities[predicted_idx]*100:.1f}%")
    print("\nAll class probabilities:")
    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        bar = '█' * int(prob * 30)
        print(f"  {cls:12s} {prob*100:5.1f}% {bar}")

def predict_from_url(url):
    """Predict class of an image from a URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img.save('temp_image.jpg')
    predict_from_file('temp_image.jpg')

if __name__ == "__main__":
    # Test with a sample image
    # Usage: python src/predict.py path/to/your/image.jpg
    if len(sys.argv) > 1:
        predict_from_file(sys.argv[1])
    else:
        print("Usage: python src/predict.py <image_path>")
        print("Example: python src/predict.py results/sample_images.png")