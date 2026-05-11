import gradio as gr
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import tensorflow as tf
from PIL import Image

print("Loading SynthCatch Models into memory...")

# --- 1. Load the Vision Transformer (PyTorch) ---
print("Loading Vision Transformer...")
vit_path = "./best_cifake_vit_model"
processor = ViTImageProcessor.from_pretrained(vit_path)
vit_model = ViTForImageClassification.from_pretrained(vit_path)

# --- 2. Load the Keras Model (TensorFlow) ---
print("Loading MobileNetV2...")
mobilenet_model = tf.keras.models.load_model('mobilenetv2_cifake.keras')

print("Models loaded successfully!")

# --- 3. The Multi-Model Prediction Logic ---
def predict_image(image, model_choice):
    if image is None:
        return None
        
    image = image.convert("RGB")
    
    # ROUTE A: Vision Transformer
    if model_choice == "Vision Transformer (SOTA Champion)":
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        results = {}
        for i, label_name in vit_model.config.id2label.items():
            results[label_name] = float(probabilities[int(i)])
        return results

    # ROUTE B: MobileNetV2
    elif model_choice == "MobileNetV2 (Transfer Learning)":
        # Shrink to 32x32 and apply the [-1 to 1] normalization it was trained on
        img_resized = image.resize((32, 32))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_array = (img_array / 127.5) - 1.0 
        
        pred = mobilenet_model.predict(img_array, verbose=0)[0][0]
        return {"FAKE": float(pred), "REAL": float(1.0 - pred)}

# --- 4. Build the Interactive Dashboard ---
print("Launching Dashboard...")
interface = gr.Interface(
    fn=predict_image,                  
    inputs=[
        gr.Image(type="pil", label="Upload Image"), 
        gr.Dropdown(
            choices=[
                "Vision Transformer (SOTA Champion)", 
                "MobileNetV2 (Transfer Learning)"
            ], 
            value="Vision Transformer (SOTA Champion)", 
            label="Select AI Model to Test"
        )
    ],       
    outputs=gr.Label(num_top_classes=2, label="AI Confidence Score"), 
    title="SynthCatch: AI Detector Dashboard",
    description="Compare how our models perform on the same image! The ViT is highly accurate, while MobileNetV2 is optimized for speed.",
    flagging_mode="never"              
)

# --- 5. Launch! ---
interface.launch(theme=gr.themes.Soft(), share=False)