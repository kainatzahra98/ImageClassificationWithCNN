import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CNN

# ── Device ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# ── Model ─────────────────────────────────────────────────
model = CNN().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss, Optimizer, Scheduler ────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# ── Training Loop ─────────────────────────────────────────
EPOCHS = 30
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

for epoch in range(EPOCHS):

    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100. * correct / total

    # Evaluation phase
    model.eval()
    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total       += labels.size(0)
            correct     += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc  = 100. * correct / total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    scheduler.step(test_acc)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}% | "
          f"Test Loss:  {test_loss:.3f} | Test Acc:  {test_acc:.1f}%")

print(f"\n✅ Training complete! Best Test Accuracy: {max(test_accs):.2f}%")

# ── Save Model ────────────────────────────────────────────
os.makedirs('model',   exist_ok=True)
os.makedirs('results', exist_ok=True)

torch.save(model.state_dict(), 'model/cnn_cifar10.pth')
print("✅ Model saved to model/cnn_cifar10.pth")

# ── Plot Training Curves ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_accs, label='Train Accuracy', color='blue',  linewidth=2)
ax1.plot(test_accs,  label='Test Accuracy',  color='green', linewidth=2)
ax1.set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 100])

ax2.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
ax2.plot(test_losses,  label='Test Loss',  color='red',  linewidth=2)
ax2.set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'CNN on CIFAR-10 | Best Test Accuracy: {max(test_accs):.1f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training curves saved to results/training_results.png")

# ── Confusion Matrix ──────────────────────────────────────
all_preds  = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix — CNN on CIFAR-10',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved to results/confusion_matrix.png")