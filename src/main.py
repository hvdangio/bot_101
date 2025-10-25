

##1. LOAD AI-model:
from transformers import pipeline
chat = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
print(chat("User: Xin chào! Bot:", max_length=100))

##2. LOAD UI:
import gradio as gr

def chatbot(msg, history):
    response = chat(msg)[0]['generated_text']
    history.append((msg, response)) #funn check
    return response
gr.ChatInterface(chatbot).launch()
