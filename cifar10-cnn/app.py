"""
Flask Web Application for CIFAR-10 CNN Image Classifier
Provides a beautiful dashboard UI with real-time image prediction
"""

import os
import io
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.model import CNN

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Setup ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_EMOJIS = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱',
    'deer': '🦌', 'dog': '🐶', 'frog': '🐸', 'horse': '🐴',
    'ship': '🚢', 'truck': '🚛'
}

# Load model
model = CNN().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'cnn_cifar10.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform pipeline (must match training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── Routes ────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result images (training curves, confusion matrix)"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    return send_from_directory(results_dir, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Create base64 thumbnail for display
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Run inference
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        # Build results
        results = []
        for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            results.append({
                'class': cls,
                'emoji': CLASS_EMOJIS[cls],
                'probability': round(prob.item() * 100, 2)
            })

        # Sort by probability (descending)
        results.sort(key=lambda x: x['probability'], reverse=True)

        predicted = results[0]

        return jsonify({
            'success': True,
            'prediction': predicted['class'],
            'emoji': predicted['emoji'],
            'confidence': predicted['probability'],
            'all_probabilities': results,
            'image': f'data:image/png;base64,{img_b64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    """Return model architecture info"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return jsonify({
        'total_parameters': f'{total_params:,}',
        'trainable_parameters': f'{trainable_params:,}',
        'device': str(device),
        'classes': CLASSES,
        'input_shape': '3 × 32 × 32',
        'architecture': {
            'block1': 'Conv2d(3→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool → Dropout(0.25)',
            'block2': 'Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool → Dropout(0.25)',
            'block3': 'Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool → Dropout(0.25)',
            'classifier': 'Flatten → Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→10)'
        },
        'metrics': {
            'test_accuracy': '86.8%',
            'training_epochs': 30,
            'optimizer': 'Adam (lr=0.001)',
            'scheduler': 'ReduceLROnPlateau'
        }
    })


if __name__ == '__main__':
    print("\n🧠 CIFAR-10 CNN Image Classifier")
    print(f"📍 Device: {device}")
    print(f"🔢 Classes: {', '.join(CLASSES)}")
    print("🌐 Starting web server...\n")
    app.run(debug=True, port=5000)
