##0. Utils....
import os
from datetime import datetime

def log_response(input_msg, output_text):
  os.makedirs("TEMP", exist_ok=True)
  with open("TEMP/bot_101.log", "a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.now()} ---\n")
    f.write(f"USER: {input_msg}\n")
    f.write(f"BOT : {output_text}\n")


##1. LOAD AI-model: ---------------
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "deepseek-ai/DeepSeek-V2-Lite"
#model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  device_map="auto",
  #torch_dtype=torch.float16  #try for performance
)
print(chat("User: Xin chào! Bot:", max_length=100))


##2. Define chatbot: ---------------
def extract_first_new_assistant_reply(generated_text):
  if "### RESPONSE: Assistant:" in generated_text:
    reply = generated_text.split("### RESPONSE: Assistant:", 1)[1]
    reply = reply.split("### ASKING: User:", 1)[0]
    reply = reply.split("### CONTEXT: User:", 1)[0]
    reply = reply.split("### ASKING: Assistant:", 1)[0]
    return reply.strip()
  return generated_text.strip()


def chatbot(msg, history):
  # Step 2.1: System prompt (role = system)
  system_prompt = {
    "role": "system",
    "content": (
      "### INSTRUCTION: Hãy trả lời thông minh, sâu sắc, hài hước, dí dỏm. Format markdown!"
    )
  }

  # Step 2.2: history (OpenAI style)
  messages = []; #messages = [system_prompt]
  for m in history[-3:]:
    if m["role"] in ("user", "assistant"):
      messages.append(m)

  # Step 2.3: prompt_text
  prompt_text = ""
  for m in messages:
    prompt_text += f"### CONTEXT: {m['role'].capitalize()}: {m['content']}\n"
  # New message from user
  prompt_text += f"### ASKING: {"user".capitalize()}: {msg}\n"
  prompt_text += "### RESPONSE: Assistant:"

  output = chat(
    prompt_text,
    max_new_tokens=300,
    temperature=0.7,
    max_length=1500  # ~1–1.5 A4 page
  )[0]["generated_text"]
  log_response(msg, output)

  # Step 3.4: Return
  generated_text = output[len(prompt_text) - len("### RESPONSE: Assistant:"):]
  bot_reply = extract_first_new_assistant_reply(generated_text)
  return {"role": "assistant", "content": bot_reply}


##3. LOAD UI: ---------------
import gradio as gr
gr.ChatInterface(
  fn=chatbot,
  chatbot=gr.Chatbot(type="messages")
).launch()

