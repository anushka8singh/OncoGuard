import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_heatmap(image_path, model_path, output_path):
    # Load model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Target layer
    target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    # Preprocess image
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Generate heatmap
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_np = np.array(img.resize((224,224))) / 255.0
    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    cv2.imwrite(output_path, heatmap)
    print("Heatmap saved:", output_path)