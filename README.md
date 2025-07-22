# LLaVA Multi-Modal Classification Framework

This repository contains a comprehensive framework for performing zero-shot and few-shot classification tasks using LLaVA (Large Language and Vision Assistant) on two different datasets: E-SNLI-VE (visual entailment) and Facebook Hateful Memes (offensive content detection).

## Overview

The framework supports:
- **Zero-shot prompting**: Direct classification without examples
- **Few-shot prompting**: Classification using in-context examples
- **Two datasets**: E-SNLI-VE for visual entailment and Facebook Hateful Memes for offensive content detection
- **Quantized inference**: 4-bit quantization for efficient GPU memory usage

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
```

## Model Configuration

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

## Output Format

Results are saved to `answer_df.csv` with the following structure:

### E-SNLI-VE Output:
```csv
img_path,hypothesis,explanation
image1.jpg,"A person is walking","Entailment. The image shows a person walking down the street."
```

### Meme Output:
```csv
img_path,explanation
meme1.png,"This meme is not offensive."
meme2.png,"This meme is offensive. Contains discriminatory language."
```
