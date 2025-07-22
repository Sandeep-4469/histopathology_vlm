# Ki-67 Histopathology Image Analysis with MedGemma

## Project Overview
This project aims to analyze Ki-67 stained histopathology images to quantify and describe cellular staining patterns, specifically for nuclear staining associated with cell proliferation. The project leverages `medgemma-4b-it`, a multimodal vision-language model from Google's MedGemma collection, fine-tuned on a custom dataset of histopathology images and annotations. The model processes images and generates structured JSON outputs describing the type of stain, presence of brown staining, stain proportion, stain level, and a detailed report.

The pipeline includes:
- **Inference**: Using the pre-trained or fine-tuned `medgemma-4b-it` model to analyze new images with k-shot learning, where examples are sourced from a CSV file.
- **Fine-Tuning**: Adapting the model to the specific dataset to improve performance on Ki-67 staining analysis.
- **Dataset**: A CSV file containing image paths and annotations (e.g., `brown_stain`, `stain_proportion`, `stain_levels`, `report`).

## Google MedGemma
The `medgemma-4b-it` model is part of Google's MedGemma collection, a family of multimodal models designed for medical applications, combining a language model (based on Gemma) with a vision encoder (SigLIP) for processing text and images. It is hosted on Hugging Face under `google/medgemma-4b-it` (access may be gated, requiring a Hugging Face API key and acceptance of Google's Health AI Developer Foundations terms). This project uses a local copy of the model stored at `/data2/sandeep/medgemma-4b-it`.

## Dataset
The dataset is stored in a CSV file at `/data2/shreyas/projects/PATHOLOGY-VLM/dataset/ki67_examples.csv`. Each row includes:
- `image`: Path to a histopathology image (e.g., `/data2/shreyas/projects/PATHOLOGY-VLM/dataset/2023-03-03/14-40X-40X-1-00DX-2023-03-03-16-07-27.jpg`).
- `brown_stain`: `"yes"` or `"no"`, indicating the presence of brown staining.
- `stain_proportion`: Float between 0.0 and 1.0, representing the proportion of stained cells.
- `stain_levels`: List of integers (e.g., `[2, 1]`), where the first integer is used as the stain level (1, 2, or 3).
- `report`: Text description of the staining pattern, cell distribution, intensity, and anomalies.

Example CSV row:
```csv
brown_stain,stain_proportion,stain_levels,report,image
yes,0.15,"[2, 1]","Ki-67 staining is present. Approximately 15% of the cells show nuclear staining in brown. The staining appears to be relatively uniform throughout the image.","/data2/shreyas/projects/PATHOLOGY-VLM/dataset/2023-03-03/81-40X-40X-1-00DX-2023-03-03-13-05-49.jpg"
```

**Note**: The current dataset contains only six examples, which may limit fine-tuning effectiveness. Expanding the dataset with more annotated images is recommended for better model performance.

## Prerequisites
- **Hardware**: A GPU with at least 24GB VRAM is recommended for fine-tuning the 4B parameter `medgemma-4b-it` model. Inference can run on less powerful hardware.
- **Environment**: Python 3.9 environment at `/data2/sandeep/ENVS/hbmeter`.
- **Dependencies**: Install required libraries:
  ```bash
  source /data2/sandeep/ENVS/hbmeter/bin/activate
  pip install transformers datasets trl peft accelerate pandas pillow
  ```
- **Model Files**: Ensure the `medgemma-4b-it` model is at `/data2/sandeep/medgemma-4b-it` with `tokenizer.json` (or `tokenizer.model` if available). Update `tokenizer_config.json` to use `"tokenizer_class": "PreTrainedTokenizerFast"` for `tokenizer.json`.

## Scripts
### 1. Inference Script (`few_shot_pathology_analysis_functions.py`)
This script performs k-shot inference on a user-provided histopathology image, using examples from the CSV to guide the model.

**Functionality**:
- Loads the model and processor from `/data2/sandeep/medgemma-4b-it`.
- Reads k examples from the CSV, excluding the target image.
- Formats examples into alternating `user`/`assistant` chat messages.
- Generates a JSON output for the target image:
  ```json
  {
    "type_of_stain": "nuclear",
    "brown_stain": "yes",
    "stain_proportion": 0.2,
    "stain_level": 2,
    "report": "The tissue sample shows Ki-67 staining with approximately 20% of cells exhibiting moderate brown nuclear staining. The staining is uniformly distributed with consistent intensity. Cells are evenly spaced, with no significant anomalies."
  }
  ```

**Usage**:
```python
from few_shot_pathology_analysis_functions import main
main(
    image_path="/data2/shreyas/projects/PATHOLOGY-VLM/dataset/2023-03-03/14-40X-40X-1-00DX-2023-03-03-16-07-27.jpg",
    k=3,
    csv_path="/data2/shreyas/projects/PATHOLOGY-VLM/dataset/ki67_examples.csv"
)
```

### 2. Fine-Tuning Script (`finetune_medgemma.py`)
This script fine-tunes the `medgemma-4b-it` model on the CSV dataset using LoRA for memory efficiency and tests the fine-tuned model.

**Functionality**:
- Loads the CSV and formats it into a Hugging Face `Dataset` with chat-style messages.
- Applies LoRA to fine-tune the model with a small learning rate (`2e-5`) and 3 epochs.
- Saves the fine-tuned model to `/data2/sandeep/medgemma-4b-it-finetuned`.
- Tests the model on a specified image, producing a JSON output similar to the inference script.

**Usage**:
```bash
python finetune_medgemma.py
```
**Expected Output**:
- Training logs every 10 steps (e.g., `{'loss': 0.5321, 'learning_rate': 1.5e-5, 'epoch': 0.5}`).
- Test output:
  ```
  🧠 Fine-Tuned Model Output:
  ```json
  {
    "type_of_stain": "nuclear",
    "brown_stain": "yes",
    "stain_proportion": 0.25,
    "stain_level": 2,
    "report": "The image shows Ki-67 staining with moderate brown nuclear staining in about 25% of the cells. The staining is relatively uniform throughout the field of view, with consistent moderate intensity."
  }
  ```
  ```

## Setup Instructions
1. **Activate Environment**:
   ```bash
   source /data2/sandeep/ENVS/hbmeter/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers datasets trl peft accelerate pandas pillow
   ```

3. **Verify Dataset**:
   - Ensure the CSV is at `/data2/shreyas/projects/PATHOLOGY-VLM/dataset/ki67_examples.csv`.
   - Check image paths:
     ```bash
     ls -l /data2/shreyas/projects/PATHOLOGY-VLM/dataset/2023-03-03/
     ```

4. **Tokenizer Configuration**:
   - Ensure `/data2/sandeep/medgemma-4b-it/tokenizer_config.json` has:
     ```json
     "tokenizer_class": "PreTrainedTokenizerFast"
     ```
   - If using `tokenizer.model`, set `"tokenizer_class": "GemmaTokenizer"` and update scripts to `use_fast=False`.

5. **Run Inference**:
   - Use the inference script for quick analysis:
     ```python
     from few_shot_pathology_analysis_functions import main
     main("/path/to/your/image.jpg", k=3)
     ```

6. **Run Fine-Tuning**:
   - Fine-tune the model to improve performance:
     ```bash
     python finetune_medgemma.py
     ```

## Troubleshooting
- **Tokenizer Error (`TypeError: not a string`)**:
  - Ensure `tokenizer.json` exists and `tokenizer_config.json` is set to `"PreTrainedTokenizerFast"`. Alternatively, obtain `tokenizer.model` and set `use_fast=False`.
- **FileNotFoundError**:
  - Verify CSV and image paths. Update `csv_path` or `image_path` as needed.
- **OutOfMemoryError**:
  - Reduce `per_device_train_batch_size` to 1 or increase `gradient_accumulation_steps` to 8 in `finetune_medgemma.py`.
- **Poor Fine-Tuning Results**:
  - The dataset is small (six examples). Add more annotated images to the CSV for better performance.

## Future Improvements
- **Expand Dataset**: Collect more Ki-67 stained images with annotations to improve fine-tuning.
- **Data Augmentation**: Apply image transformations (e.g., rotation, flipping) to increase dataset diversity.
- **Hyperparameter Tuning**: Adjust `learning_rate`, `num_train_epochs`, or LoRA rank (`r`) based on performance.
- **Multi-GPU Training**: Use `accelerate` for distributed training if multiple GPUs are available.

## Contact
For issues or enhancements, contact the project maintainer or refer to the Hugging Face documentation for `medgemma-4b-it` and the `transformers` library.
