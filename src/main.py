import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


##1. LOAD AI-model: ---------------
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "deepseek-ai/DeepSeek-V2-Lite"
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  device_map="cpu",  #no GPU now
  #torch_dtype=torch.float16  #try for performance
)
print(chat("User: Xin chào! Bot:", max_length=100))


##2. Define chatbot: ---------------
def chatbot(msg, history):
  full_prompt = msg #no_history
  output = chat(full_prompt, max_new_tokens=50, do_sample=True
                , temperature=0.7, max_length=200
                )[0]["generated_text"]
  # history = history + [(msg, output)]
  return output


##3. LOAD UI: ---------------
import gradio as gr
gr.ChatInterface(chatbot).launch()
