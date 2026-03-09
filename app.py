import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def predict(text):
    result = classifier(text)
    return result[0]['label']

interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis App"
)

interface.launch()