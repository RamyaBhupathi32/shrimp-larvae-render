import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load trained model
model = YOLO("best.pt")

def count_shrimp(image):
    if image is None:
        return None, "No image uploaded"

    image_np = np.array(image)
    results = model(image_np)

    count = len(results[0].boxes)
    result_image = results[0].plot()

    return result_image, f"Detected larvae: {count}"

demo = gr.Interface(
    fn=count_shrimp,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(), gr.Textbox()],
    title="Shrimp Larvae Counter"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
