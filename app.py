from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

from transformers import pipeline

pipe = pipeline(model="carlosaccardi/whisper-swedish-new")

def transcribe(audio):
    text = pipe(audio)["text"]
    generator = pipeline('text-generation', model = 'birgermoell/swedish-gpt')
    return generator(text, max_length = 50, num_return_sequences=1)[0]['generated_text']

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Swedish text generation",
    description="Tell the API a sentence to be completed.",
)

iface.launch()