# %%
from huggingface_hub import login
login("hf_fJDqtGfzdgfdfbvzfbzdfvbzfdvzfvzdfvz")

# %%
# !pip install transformers
# !pip install peft
# !pip install accelerate
# !pip install -U bitsandbytes

# %%
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

# %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("bhutumbanerjee/esnli-ve")

# print("Path to dataset files:", path)

# %%
# 4 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# %%
# Load model and Data
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448", quantization_config=bnb_config, torch_dtype = torch.bfloat16, device_map='auto')

# %%
df = pd.read_csv("/home/paritosh/soham/snli_ve_fin/snli_ve_train_fin.csv")

# %%
# For ESNLI VE Dataset
class CustomDataset(Dataset):
    def __init__(self, df):
        self.dataset = df

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        img_path = item["image_path"]
        hypothesis = item["hypothesis"]
        explanation = item["exp"]

        image = Image.open(os.path.join("/home/paritosh/.cache/kagglehub/datasets/bhutumbanerjee/esnli-ve/versions/1/imgs/imgs/", img_path))

        # Format text WITHOUT <image> token - processor will add it
        input_text = f"explain en {hypothesis}"
        
        # Processor automatically adds <image> token at the beginning
        inputs = processor(
            text=input_text,
            images=image,
            # text_target=explanation,
            return_tensors="pt"
        )

        # Get input sequence length
        input_length = inputs['input_ids'].shape[1]
        
        # The processor internally creates: "<image>explain en {hypothesis}"
        
        # Process TARGET using tokenizer only (NO IMAGE NEEDED)
        target_tokens = processor.tokenizer(
            text=explanation,
            add_special_tokens=False,
            return_tensors="pt"
        )['input_ids'].squeeze(0)

        # Create labels with same length as input
        labels = torch.full((input_length,), -100, dtype=torch.long)

        # Fill the end with target tokens
        target_length = min(len(target_tokens), input_length)
        labels[-target_length:] = target_tokens[:target_length]        
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': labels
        }       


# %%
ds = CustomDataset(df)

# %%
def collate_fn(batch):
    print(f"Batch received in collate_fn with {len(batch)} items")
    
    # Extract components
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Create batch dictionary
    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'pixel_values': pixel_values,
        'labels': labels
    }
    
    # Move to model device
    model_device = next(model.parameters()).device
    for key, value in batch_dict.items():
        if torch.is_floating_point(value):
            batch_dict[key] = value.to(dtype=torch.bfloat16, device=model_device)
        else:
            batch_dict[key] = value.to(device=model_device)

    # CHECK DEVICES AFTER MOVING:
    print("📱 Devices after moving:")
    for key, value in batch_dict.items():
        print(f"  {key}: {value.device}")
    
    return batch_dict

# %%
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

lora_config = LoraConfig(
    r=64,
    lora_alpha=256,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# %%
"/home/paritosh/.cache/kagglehub/datasets/bhutumbanerjee/esnli-ve/versions/1/imgs/imgs/134206.jpg"

# %%
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
# from torch.utils.data import DataLoader

# loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

# for batch in loader:
#     for k, v in batch.items():
#         print(k, v.shape)
#     break

# %%
from transformers import TrainingArguments
args = TrainingArguments(
    num_train_epochs=2,
    logging_first_step=True,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    learning_rate=2e-4,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=1,
    optim="adamw_torch",
    save_strategy="epoch",
    save_total_limit=1,
    output_dir="/home/paritosh/soham/save_checkpoin",
    bf16=True,
    report_to=None,
    dataloader_pin_memory=False
)

# %%
from transformers import Trainer

trainer = Trainer(
        model=model,
        train_dataset=ds ,
        data_collator=collate_fn,
        args=args
        )

# %%
trainer.train()

# %%
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Set device
device = model.device

# Define DataLoader
train_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Set model to train mode
model.train()

# Number of epochs
num_epochs = 2

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    total_loss = 0

    for step, batch in enumerate(tqdm(train_loader)):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if (step + 1) % 8 == 0:  # gradient_accumulation_steps = 8
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        # Optional: Print loss every step
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# %%


# %%


# %%



