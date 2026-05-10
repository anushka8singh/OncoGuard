import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

def main():

    # ==============================
    # DEVICE
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ==============================
    # TRANSFORMS (LIGHTWEIGHT)
    # ==============================
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ==============================
    # DATASET
    # ==============================
    train_data = datasets.ImageFolder("../data/brain/train", transform=train_transform)
    test_data = datasets.ImageFolder("../data/brain/test", transform=test_transform)

    print("Classes:", train_data.classes)

    # ==============================
    # DATALOADER (SAFE)
    # ==============================
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=0)

    # ==============================
    # MODEL
    # ==============================
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # 4 classes
    model.fc = nn.Linear(model.fc.in_features, 4)

    model = model.to(device)

    # ==============================
    # LOSS + OPTIMIZER
    # ==============================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # ==============================
    # TRAINING (STAGE 1)
    # ==============================
    print("\n--- Stage 1 Training ---")
    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ==============================
    # FINE-TUNING (STAGE 2)
    # ==============================
    print("\n--- Fine-Tuning ---")

    for param in model.layer4.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"FineTune Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================
    torch.save(model.state_dict(), "../models/brain_model.pth")
    print("\n✅ Brain model saved!")

# ==============================
# ENTRY POINT (IMPORTANT FOR WINDOWS)
# ==============================
if __name__ == "__main__":
    main()