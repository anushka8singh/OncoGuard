from gradcam import generate_heatmap

image_path = "../data/brain/test/glioma/Te-gl_1.jpg"
model_path = "../models/brain_model.pth"
output_path = "../heatmap.jpg"

generate_heatmap(image_path, model_path, output_path)