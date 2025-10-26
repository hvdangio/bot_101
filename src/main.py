##0. Utils....
import os
from datetime import datetime

def log_response(input_msg, output_text):
  os.makedirs("TEMP", exist_ok=True)
  with open("TEMP/bot_101.log", "a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.now()} ---\n")
    f.write(f"USER: {input_msg}\n")
    f.write(f"BOT : {output_text}\n")

def clean_bot_reply(output_text):
  parts = output_text.split("Assistant:")
  if len(parts) > 1:
    return parts[-1].strip()
  return output_text.strip()


##1. LOAD AI-model: ---------------
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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
  # Step 2.1: System prompt (role = system)
  system_prompt = {
    "role": "system",
    "content": (
      "Bạn là một trợ lý AI thông minh, biết suy luận logic và trả lời một cách hài hước, duyên dáng. "
      "Hãy thêm chút hài hước nếu thích hợp, nhưng vẫn giữ nội dung chính xác. "
      "Nếu không biết câu trả lời, cứ từ chối nhẹ nhàng chứ đừng bịa nhé!"
    )
  }

  # Step 2.2: history (OpenAI style)
  messages = [system_prompt]
  for m in history[-3:]:
    if m["role"] in ("user", "assistant"):
      messages.append(m)

  # New message from user
  messages.append({"role": "user", "content": msg})

  # Step 2.3: prompt_text
  prompt_text = ""
  for m in messages:
    prompt_text += f"{m['role'].capitalize()}: {m['content']}\n"
  prompt_text += "Assistant:"

  output = chat(
    prompt_text,
    max_new_tokens=300,
    temperature=0.7,
    max_length=1500  # ~1–1.5 A4 page
  )[0]["generated_text"]
  log_response(msg, output)

  # Step 3.4: Return
  bot_reply = clean_bot_reply(output)
  return {"role": "assistant", "content": bot_reply}


##3. LOAD UI: ---------------
import gradio as gr
gr.ChatInterface(
  fn=chatbot,
  chatbot=gr.Chatbot(type="messages")
).launch()

