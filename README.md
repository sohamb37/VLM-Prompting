# LLaVA Multi-Modal Classification Framework

This repository contains a comprehensive framework for performing zero-shot and few-shot classification tasks using LLaVA (Large Language and Vision Assistant) on two different datasets: E-SNLI-VE (visual entailment) and Facebook Hateful Memes (offensive content detection). Additionally, it includes PaliGemma fine-tuning capabilities for enhanced performance on the E-SNLI-VE dataset.

## Overview

The framework supports:
- **Zero-shot prompting**: Direct classification without examples
- **Few-shot prompting**: Classification using in-context examples
- **Two datasets**: E-SNLI-VE for visual entailment and Facebook Hateful Memes for offensive content detection
- **Quantized inference**: 4-bit quantization for efficient GPU memory usage
- **PaliGemma fine-tuning**: Custom fine-tuning on E-SNLI-VE dataset with LoRA adapters
- **Fine-tuned model inference**: Inference using fine-tuned PaliGemma models

## Usage

### Basic Usage

Run zero-shot classification on meme dataset:
```bash
python llava_classification.py --dataset meme --shot_type 0_shot
```

Run few-shot classification on E-SNLI-VE dataset:
```bash
python llava_classification.py --dataset esnli --shot_type few_shot
```

### Command Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | `esnli`, `meme` | `meme` | Dataset to evaluate on |
| `--shot_type` | `0_shot`, `few_shot` | `0_shot` | Prompting strategy |

### Examples

```bash
# Zero-shot visual entailment
python llava_classification.py --dataset esnli --shot_type 0_shot

# Few-shot offensive content detection
python llava_classification.py --dataset meme --shot_type few_shot

# PaliGemma fine-tuning
python paligemma_finetune.py

# PaliGemma inference
python paligemma_inference.py
```

## Model Configuration

### LLaVA Model Configuration

The framework uses LLaVA-v1.6-Mistral-7B with the following configuration:

```python
# 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Model loading
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    quantization_config=bnb_config, 
    device_map="auto"
)
```

### PaliGemma Model Configuration

The PaliGemma fine-tuning uses the following configuration:

```python
# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Model and processor loading
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-448", 
    quantization_config=bnb_config, 
    torch_dtype=torch.bfloat16, 
    device_map='auto'
)
```

## Tasks Description

### 1. E-SNLI-VE (Visual Entailment)
- **Task**: Determine if a hypothesis entails, contradicts, or is neutral to an image
- **Classes**: `entailment`, `contradiction`, `neutral`
- **Output**: Classification + explanation

**Zero-shot prompt example:**
```
Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis 
'A person is riding a bicycle' follows the image, contradicts it, or is 
neutral to it. Also give a 1-line explanation for your answer.
```

### 2. Facebook Hateful Memes (Offensive Content Detection)
- **Task**: Detect if a meme contains offensive content
- **Classes**: `offensive`, `not offensive`
- **Output**: Classification + explanation (if offensive)

**Zero-shot prompt example:**
```
Is this image offensive? If it is offensive, give a single line explanation. 
Otherwise, simply state that it is 'not offensive'.
```

## Few-Shot Learning

### E-SNLI-VE Few-Shot
- Uses 3 examples (one for each class: entailment, contradiction, neutral)
- Examples are randomly sampled from training set
- Each example includes image + hypothesis + explanation

### Meme Few-Shot
- Uses 2 examples (one offensive, one not offensive)
- Examples are sampled from correctly classified training instances
- Provides context for offensive content detection

## PaliGemma Fine-Tuning

### Overview
The framework includes fine-tuning capabilities for PaliGemma on the E-SNLI-VE dataset using LoRA (Low-Rank Adaptation) for parameter-efficient training.

### Fine-Tuning Configuration

#### LoRA Configuration
```python
lora_config = LoraConfig(
    r=64,                                    # Rank of adaptation
    lora_alpha=256,                         # LoRA scaling parameter
    target_modules=find_all_linear_names(model),  # Target all linear layers
    lora_dropout=0.1,                       # Dropout for LoRA layers
    bias="none",                           # No bias adaptation
    task_type="CAUSAL_LM"                  # Causal language modeling
)
```

#### Training Arguments
```python
args = TrainingArguments(
    num_train_epochs=2,                    # Number of training epochs
    per_device_train_batch_size=1,         # Batch size per device
    gradient_accumulation_steps=8,         # Gradient accumulation
    learning_rate=2e-4,                    # Learning rate
    weight_decay=1e-6,                     # Weight decay
    warmup_steps=2,                        # Warmup steps
    bf16=True,                            # Use bfloat16 precision
    save_strategy="epoch",                 # Save after each epoch
    output_dir="/path/to/checkpoints"      # Checkpoint directory
)
```

### Dataset Format for Fine-Tuning
The fine-tuning expects a CSV file with the following columns:
- `image_path`: Path to the image file
- `hypothesis`: The hypothesis text to evaluate
- `exp`: The explanation/target text for training

It is important to note that the csv file is created by considering the samples where a classifier predicted output is same as the ground-truth. The purpose of the VLM is to explain the working of the classifier.

### Custom Dataset Class
The framework includes a custom dataset class that:
- Processes images and text using PaliGemma processor
- Formats input text as: `"explain en {hypothesis}"`
- Creates proper labels for training with explanation targets
- Handles padding and tokenization automatically

### Fine-Tuning Process
1. **Data Loading**: Load training data from CSV
2. **Model Preparation**: Apply LoRA configuration to base model
3. **Dataset Creation**: Create custom dataset with proper formatting
4. **Training**: Use HuggingFace Trainer with custom collate function
5. **Checkpoint Saving**: Save LoRA adapters at specified intervals

## PaliGemma Inference

### Loading Fine-Tuned Model
```python
# Load base model
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-448", 
    quantization_config=bnb_config, 
    torch_dtype=torch.bfloat16, 
    device_map='auto'
).eval()

# Load LoRA adapters
lora_ckp_dir = "/path/to/checkpoint"
lora_model = PeftModel.from_pretrained(model, lora_ckp_dir)
```

### Inference Process
The inference process includes:

1. **Text Preprocessing**: Format hypothesis text for the model
2. **Generation**: Use the fine-tuned model to generate explanations
3. **Post-processing**: Extract and clean generated responses
4. **Memory Management**: Clear GPU cache between inferences

### Generation Parameters
```python
output = lora_model.generate(
    **inputs,
    max_new_tokens=32,           # Maximum tokens to generate
    temperature=1.2,             # Sampling temperature
    repetition_penalty=1.5,      # Reduce repetition
    top_p=0.9,                  # Nucleus sampling
    top_k=50                    # Top-k sampling
)
```



## Output Format

#### E-SNLI-VE Output:
```csv
img_path,hypothesis,explanation
image1.jpg,"A person is walking","Entailment. The image shows a person walking down the street."
```

#### Meme Output:
```csv
img_path,explanation
meme1.png,"This meme is not offensive."
meme2.png,"This meme is offensive. Contains discriminatory language."
```

## Results

| Dataset | Models | BLEU | BERTScore |
|----------|---------|---------|-------------|
| `memes` | `LLaVA (0shot)` | `0.01` | `0.894` |
| `memes` | `LLaVA (2shot)` | `0.01` | `0.864` |
| `esnli-ve` | `LLaVA (0shot)` | `0.02` | `0.876` |
| `esnli-ve` | `LLaVA (3shot)` | `0.03` | `0.869` |
| `esnli-ve` | `PaLiGemma Ft` | `0.17` | `0.894` |


## Conclusion

Direct fine-tuning and prompting strategies show limited effectiveness for explanation generation, highlighting the need for dedicated frameworks specifically designed for generating explanations in natural language.

## Performance Notes

- **4-bit Quantization**: Significantly reduces memory usage with minimal performance impact
- **LoRA Fine-tuning**: Parameter-efficient approach that only trains ~1-5% of model parameters
- **Gradient Accumulation**: Enables larger effective batch sizes on limited GPU memory
- **Memory Management**: Automatic GPU cache clearing prevents memory overflow during inference
