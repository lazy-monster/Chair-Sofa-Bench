import gradio as gr
from fastai.vision.all import *

categories = ["Bench", "Chair", "Sofa"]
learn = load_learner("model.pkl")


def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


# Updated components
image = gr.Image(type="pil", shape=(192, 192))
label = gr.Label()
examples = ["chair1.jpg", "bench1.jpg", "sofa1.jpg", "dunno1.jpg"]

# Interface with new components
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()
