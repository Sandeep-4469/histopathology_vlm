from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import os
import pandas as pd
import ast

def load_model_and_processor(model_path):
    """Load the MedGemma model and processor."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Check tokenizer files
    tokenizer_model = os.path.join(model_path, "tokenizer.model")
    tokenizer_json = os.path.join(model_path, "tokenizer.json")
    use_fast = os.path.exists(tokenizer_json) and not os.path.exists(tokenizer_model)

    if not (os.path.exists(tokenizer_model) or os.path.exists(tokenizer_json)):
        raise FileNotFoundError(f"Neither tokenizer.model nor tokenizer.json found in {model_path}. Ensure the correct tokenizer file is present.")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=use_fast
        )
        return model, processor
    except Exception as e:
        raise Exception(f"Failed to load model or processor from {model_path}. If using tokenizer.json, set tokenizer_class to 'PreTrainedTokenizerFast' in tokenizer_config.json. If using tokenizer.model, ensure it exists. Error: {str(e)}")

def load_csv_examples(csv_path, target_image_path, k):
    """Load and parse k examples from CSV, excluding the target image."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Verify the path.")
    except Exception as e:
        raise Exception(f"Failed to load CSV file: {str(e)}")

    # Ensure required columns exist
    required_columns = ["brown_stain", "stain_proportion", "stain_levels", "report", "image"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")

    # Filter out the target image
    df = df[df["image"] != target_image_path]

    # Select up to k examples
    k = max(1, min(k, len(df)))
    df = df.head(k)

    example_images = []
    example_outputs = []
    for _, row in df.iterrows():
        image_path = os.path.abspath(row["image"])
        try:
            example_images.append(Image.open(image_path).convert("RGB"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Example image not found at {image_path}. Verify the path.")

        # Parse stain_levels
        try:
            stain_levels = ast.literal_eval(row["stain_levels"])
            stain_level = stain_levels[0] if isinstance(stain_levels, list) and len(stain_levels) > 0 else 2
        except (ValueError, SyntaxError):
            stain_level = 2

        # Generate JSON output
        example_output = f"""```json
{{
  "type_of_stain": "nuclear",
  "brown_stain": "{row["brown_stain"]}",
  "stain_proportion": {row["stain_proportion"]},
  "stain_level": {stain_level},
  "report": "{row["report"]}"
}}
```"""
        example_outputs.append(example_output)

    return example_images, example_outputs

def load_target_image(image_path):
    """Load the target image."""
    try:
        return Image.open(os.path.abspath(image_path)).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Target image not found at {image_path}. Verify the path.")

def prepare_messages(example_images, example_outputs, target_image):
    """Prepare chat messages with few-shot examples and target image."""
    system_prompt = """You are a pathology assistant specialized in analyzing stained histopathology images, particularly those with Ki-67 staining."""

    analysis_instruction = """Analyze the provided histopathology image and return your findings in the following JSON format, inside markdown triple backticks:
```json
{
  "type_of_stain": "nuclear" or "cytoplasmic",
  "brown_stain": "yes" or "no",
  "stain_proportion": float (between 0.0 and 1.0),
  "stain_level": integer (1, 2, or 3),
  "report": "detailed report as a remark, describing your interpretation like a medical expert. Mention cell distribution, staining intensity, patterns, and any anomalies. Avoid repeating field names."
}
```"""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
    ]

    for i in range(len(example_images)):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": analysis_instruction},
                {"type": "image", "image": example_images[i]}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example_outputs[i]}]
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": analysis_instruction},
            {"type": "image", "image": target_image}
        ]
    })

    return messages

def run_inference(model, processor, messages):
    """Run model inference and decode output."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
    except Exception as e:
        raise Exception(f"Failed to preprocess inputs. Ensure the model supports multiple images in a single conversation. Error: {str(e)}")

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

def main(image_path, k=3, csv_path="dataset/2023-03-03.csv"):
    """Main function to analyze histopathology image with k-shot learning."""
    model_path = "/data2/sandeep/medgemma-4b-it"
    model, processor = load_model_and_processor(model_path)
    example_images, example_outputs = load_csv_examples(csv_path, image_path, k)
    target_image = load_target_image(image_path)
    messages = prepare_messages(example_images, example_outputs, target_image)
    output = run_inference(model, processor, messages)
    print("\n Model Output:\n", output)

if __name__ == "__main__":
    # Example usage
    image_path = "dataset/2023-03-03/77-40X-40X-1-00DX-2023-03-03-12-37-56.jpg"
    k = 3
    main(image_path, k)