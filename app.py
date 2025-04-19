import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Initialize the main app window
app = tk.Tk()
app.geometry("532x632")
app.title("DiffuseMate")

# Set UI appearance to dark mode
ctk.set_appearance_mode("dark")

# Entry widget for the text prompt
prompt_entry = ctk.CTkEntry(
    height=40,
    width=512,
    text_font=("Arial", 20),
    text_color="black",
    fg_color="white"
)
prompt_entry.place(x=10, y=10)

# Label to display the generated image
image_label = ctk.CTkLabel(height=512, width=512)
image_label.place(x=10, y=110)

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=auth_token
).to(device)

# Generate function that creates an image from the prompt
def generate_image():
    with autocast(device):
        result = pipe(prompt_entry.get(), guidance_scale=8.5)
        image = result["sample"][0]

    # Save and display the image
    image.save("generatedimage.png")
    tk_image = ImageTk.PhotoImage(image)
    image_label.configure(image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

# Button to trigger image generation
generate_button = ctk.CTkButton(
    height=40,
    width=120,
    text_font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    command=generate_image,
    text="Generate"
)
generate_button.place(x=206, y=60)

# Start the app
app.mainloop()
