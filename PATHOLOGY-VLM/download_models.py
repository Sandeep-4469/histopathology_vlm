import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from huggingface_hub import login

# Your Hugging Face token for gated models
HF_TOKEN = ""  # ← INSERT YOUR TOKEN HERE
login(token=HF_TOKEN)

# Target local save directory
save_dir = "/data2/sandeep/medgemma-4b-it"

# Check disk space (~12 GB required)
required_space = 12 * 1024 * 1024 * 1024  # 12 GB in bytes
stat = os.statvfs(save_dir)
available_space = stat.f_bavail * stat.f_frsize
if available_space < required_space:
    raise RuntimeError(f"Insufficient disk space in {save_dir}. Required: 12 GB, Available: {available_space / (1024**3):.2f} GB")

try:
    model_id = "google/medgemma-4b-it"

    print("🔤 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    print("🧠 Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN)

    print("🖼️  Downloading processor...")
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)

    print(f"💾 Saving all components to {save_dir}...")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print("✅ All components successfully saved for offline use!")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("🔑 Ensure you've accepted the model license at: https://huggingface.co/google/medgemma-4b-it")
    print("🧩 Also verify disk space and folder permissions.")