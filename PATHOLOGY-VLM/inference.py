from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_path = "/data2/sandeep/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True  # Ensures 100% offline
)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# === Load image from disk ===
image_path = "dataset/2023-03-03/14-40X-40X-1-00DX-2023-03-03-16-07-27.jpg"
image = Image.open(image_path).convert("RGB")

user_prompt = """You are a pathology assistant specialized in analyzing stained histopathology images.

Please analyze the provided image and return your findings in the following JSON format, inside markdown triple backticks:

```json
{
  "type_of_stain": "nuclear" or "cytoplasmic",
  "brown_stain": "yes" or "no",
  "stain_proportion": float (between 0.0 and 1.0),
  "stain_level": integer (1, 2, or 3),
  "report": "detailed report as a paragraph, describing your interpretation like a medical expert. Mention cell distribution, staining intensity, patterns, and any anomalies. Avoid repeating field names."
}
"""

# === Prepare messages ===
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a medical vision model specialized in analyzing stained tissue images."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image}
        ]
    }
]

# === Preprocess ===
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# === Generate ===
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )
    generation = generation[0][input_len:]

# === Decode output ===
decoded = processor.decode(generation, skip_special_tokens=True)
print("\n🧠 Model Output:\n", decoded)
