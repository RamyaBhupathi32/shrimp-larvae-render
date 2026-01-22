import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

# Load model ONCE when server starts
model = YOLO("best.pt")

def count_shrimp(image):
    if image is None:
        return None, "No image uploaded"

    # Convert PIL image → numpy
    img = np.array(image)

    # Resize to make CPU inference fast (VERY IMPORTANT)
    img = cv2.resize(img, (224, 224))

    # Run YOLO
    results = model(img)

    # Count detections
    count = len(results[0].boxes)

    # Get annotated image
    output_img = results[0].plot()

    return output_img, f"Detected larvae: {count}"

demo = gr.Interface(
    fn=count_shrimp,
    inputs=gr.Image(type="pil", label="Upload Shrimp Image"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Larvae Count")
    ],
    title="Shrimp Larvae Counter",
    description="YOLOv8-based shrimp larvae detection"
)

# IMPORTANT FOR RENDER
demo.launch(server_name="0.0.0.0", server_port=7860)
