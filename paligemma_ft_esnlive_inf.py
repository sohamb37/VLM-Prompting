# %%
df_test = pd.read_csv("/kaggle/input/esnli-ve/test_processed.csv")

# %%
from huggingface_hub import login
login("hf_dfvsfvzdfzdfvzdfvzdff")

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
# 4 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# %%
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
model_id = "google/paligemma-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448", quantization_config=bnb_config, torch_dtype = torch.bfloat16, device_map='auto').eval()

# %%
lora_ckp_dir = "/kaggle/working/paligemma/checkpoint-1125"

# %%
lora_model = PeftModel.from_pretrained(model, lora_ckp_dir)

# %%
device = torch.device("cuda")

# %%
lora_model.to(device)

# %%
#Inference without batching
finetuned_responses = []
final_image_paths = []
for i in range(0, len(df_test)):
    prompt = df_test["hypothesis"][i]
    raw_image = Image.open(os.path.join("/kaggle/input/esnli-ve/imgs/imgs", df_test["Flickr30kID"][i])).convert("RGB")
    
    #preprocessing
#     raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    
    # Generate output from the model
    output = lora_model.generate(**inputs,
                            max_new_tokens=32,           # Increase token count
                            temperature=1.2,              # Encourage more creative/verbose output
                            repetition_penalty=1.5,       # Reduce repetition
                            top_p=0.9,                    # Control diversity
                            top_k=50                      # Control top-k sampling
                        )

    # Decode the output and print the generated text
    response = processor.decode(output[0], skip_special_tokens=True)
    print(response, i+1)
    finetuned_responses.append(response)
    final_image_paths.append(os.path.join("/kaggle/input/esnli-ve/imgs/imgs", df_test["Flickr30kID"][i]))
    
    #manually clear gpu
    del inputs
    torch.cuda.empty_cache()

# %%
final_output = []
final_output = [response.split("\n")[1] for response in finetuned_responses]


#saving the outputs
final_df_save = pd.DataFrame({"img_path": final_image_paths, "palgm_ft_output": final_output})
final_df_save.to_csv("/kaggle/working/paligemma_ft_snliVE.csv")

# %%


# %%


# %%


# %%



