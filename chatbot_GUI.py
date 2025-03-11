import random
import json
import torch
import tkinter as tk
from tkinter import scrolledtext
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Faisal"

def chat(event=None):
    user_input = entry.get()
    if not user_input.strip():
        return
    if user_input.lower() == "quit":
        root.destroy()
        return
    
    entry.delete(0, tk.END)
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"You: {user_input}\n", "user")
    chat_box.yview(tk.END)
    
    sentence = tokenize(user_input)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)
    
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                chat_box.insert(tk.END, f"{bot_name}: {response}\n", "bot")
    else:
        chat_box.insert(tk.END, f"{bot_name}: I do not understand...\n", "bot")
    
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)

# GUI Setup
root = tk.Tk()
root.title("Faisal Chatbot")
root.geometry("500x500")
root.configure(bg="#222831")

# Chatbox
chat_box = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD, state=tk.DISABLED, bg="#eeeeee", fg="#222831", font=("Arial", 12))
chat_box.pack(pady=10, padx=10)
chat_box.tag_config("user", foreground="#007ACC", font=("Arial", 12, "bold"))
chat_box.tag_config("bot", foreground="#00A86B", font=("Arial", 12))

# Display Welcome Message
chat_box.config(state=tk.NORMAL)
chat_box.insert(tk.END, f"{bot_name}: Hello, welcome to Faisal's Chatbot! Learn more about him by asking questions.\n", "bot")
chat_box.config(state=tk.DISABLED)

# Entry Field
entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=5, padx=10)
entry.bind("<Return>", chat)  # Bind Enter key

# Send Button
send_button = tk.Button(root, text="Send", command=chat, font=("Arial", 12, "bold"), bg="#00ADB5", fg="white", relief=tk.RAISED)
send_button.pack(pady=10)

# Run GUI
entry.focus()
root.mainloop()