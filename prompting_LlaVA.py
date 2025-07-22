# %%
import numpy as np
import pandas as pd
import torch
import random
import accelerate
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaNextForConditionalGeneration
import requests
from PIL import Image
import os
import argparse

# %%
# Function to seed all random processes
def seed_all(seed=42):    
    print(f"[ Using Seed : {seed} ]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# %%
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )

# %%
# Load the model in half-precision
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config = bnb_config, device_map = "auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# %%
def zero_shot_esnli_classification(df_test, image_dir, seed=42):
    seed_all(seed)
    results = []
    
    for i, row in df_test.iterrows():
        hypothesis = row["hypothesis"]
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis '{hypothesis}' follows the image, contradicts it, or is neutral to it. Also give a 1-line explanation for your answer."}
            ]
        }]
        
        image = Image.open(os.path.join(image_dir, row["Flickr30kID"])).convert('RGB')
        prompt = processor.apply_chat_template(conversation, add_generation_prompt = True)
        inputs = processor(images=image, text=prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=True)

        # response = generate_response(image, conversation)
        response = processor.decode(output[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response
        results.append(response)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df_test)} samples")
    
    return results


# %%
def select_gold_label_img(seed = 42):
    seed_all(seed)
    import pandas as pd
    df = pd.read_csv("/kaggle/input/esnli-ve/train_processed.csv")
    entailment = df[df["gold_label"] == "entailment"].sample(1)
    entailment_img = entailment["Flickr30kID"].values[0]
    entailment_hypothesis = entailment["hypothesis"].values[0]
    entailment_exp = entailment["explanation"].values[0]
    contradiction = df[df["gold_label"] == "contradiction"].sample(1)
    contradiction_img = contradiction["Flickr30kID"].values[0]
    contradiction_hypothesis = contradiction["hypothesis"].values[0]
    contradiction_exp = contradiction["explanation"].values[0]
    neutral = df[df["gold_label"] == "neutral"].sample(1)
    neutral_img = neutral["Flickr30kID"].values[0]
    neutral_hypothesis = neutral["hypothesis"].values[0] 
    neutral_exp = neutral["explanation"].values[0]
    return entailment_img, entailment_hypothesis, entailment_exp, contradiction_img, contradiction_hypothesis, contradiction_exp, neutral_img, neutral_hypothesis, neutral_exp

# %%
def few_shot_esnli_classification(df_test, image_dir, seed=42):
    seed_all(seed)
    import os
    results_3 = []

    for i, row in df_test.iterrows():
        # hypothesis = row["hypothesis"]
        # conversation = 
        ent_img, ent_hyp, ent_exp, con_img, con_hyp, con_exp, neu_img, neu_hyp, neu_exp = select_gold_label_img()

        ent_img = Image.open(f"/kaggle/input/esnli-ve/imgs/imgs/{ent_img}")
        con_img = Image.open(f"/kaggle/input/esnli-ve/imgs/imgs/{con_img}")
        neu_img = Image.open(f"/kaggle/input/esnli-ve/imgs/imgs/{neu_img}")

        hypothesis = df_test["hypothesis"][i]

        conversation = [
            
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # entailment image and hypothesis
                    {"type": "text", "text": f"Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis that {ent_hyp} follows the image or contradicts it or is neutral to it. Also give 1 line explanation for your answer."},
                ],
            },       
            {
                "role": "assistant",
                "content": [                    
                    {"type": "text", "text":  f"Entailment. {ent_exp}"},
                ],
            },        
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # contradiction image and hypothesis
                    {"type": "text", "text": f"Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis that {con_hyp} follows the image or contradicts it or is neutral to it. Also give 1 line explanation for your answer."},
                ],
            },
            {
                "role": "assistant",
                "content": [                    
                    {"type": "text", "text":  f"Contradiction. {con_exp}"},
                ],
            }, 
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # neutral image and hypothesis
                    {"type": "text", "text": f"Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis that {neu_hyp} follows the image or contradicts it or is neutral to it. Also give 1 line explanation for your answer."},
                ],
            },
            {
                "role": "assistant",
                "content": [                    
                    {"type": "text", "text":  f"Neutral. {neu_exp}"},
                ],
            }, 
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Image prompt
                    {"type": "text", "text":  f"Answer with 'entailment', 'contradiction' or 'neutral' if the hypothesis that {hypothesis} follows the image or contradicts it or is neutral to it. Also give 1 line explanation for your answer."},
                ],
            },   
            
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt = True)
        image = Image.open(os.path.join(image_dir, df_test["Flickr30kID"][i]))
        inputs = processor(text = prompt, images = [ent_img, con_img, neu_img, image], padding=True, return_tensors="pt").to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=50, temperature = 0.2, top_k = 5, top_p = 0.9, do_sample = True)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(output, i)

        results_3.append(output)
    print("done")
    return results_3

# %%
# def generate_response(image, conversation, max_new_tokens):
#     """ Function to generate the response for batch size = 1"""

#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(images=image, text=prompt, return_tensors='pt')
        
#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
#     response = processor.decode(output[0], skip_special_tokens=True)
#     return response.split("[/INST]")[-1].strip() if "[/INST]" in response else response

# %%
def zero_shot_meme_classification(df_test, image_dir, seed=42):
    """0 shot meme classification and track the progress"""

    seed_all(seed)
    results = []
    
    for i, row in df_test.iterrows():
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this image offensive? If it is offensive, give a single line explanation. Otherwise, simply state that it is 'not offensive'."},
                {"type": "image"},
            ]
        }]
        
        image_path = row.get('img_path')
        image = Image.open(os.path.join(image_dir, image_path))
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=True)

        response = processor.decode(output[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response
        # response = generate_response(image, conversation, max_new_tokens=64)
        results.append(response)
        
        if (i + 1) % 100 == 0:    
            print(f"Processed {i + 1}/{len(df_test)} images")
    
    return results

# %%
def select_offensive_inoffensive_image(seed=42):
    seed_all(seed)
    import pandas as pd
    df = pd.read_csv("/kaggle/input/train-prompt/train_prompt (1).csv")
    zero_matches = df[(df['gt_label'] == 0) & (df['pred_label'] == 0)]
    one_matches = df[(df['gt_label'] == 1) & (df['pred_label'] == 1)]

    not_offensive_image = zero_matches['img_path'].sample(1)
    
    sampled_offensive = one_matches.sample(1)
    offensive_image = sampled_offensive["img_path"]
    offensive_explanation = sampled_offensive['explanation']
    #     print(offensive_explanation.iloc[0])
    return not_offensive_image.iloc[0], offensive_image.iloc[0], offensive_explanation.iloc[0]

# %%
def few_shot_meme_classification(df_test, image_dir, seed=42):
    seed_all(seed)
    import os
    from PIL import Image

    results_2 = []

    for i in range(len(df_test)):
        not_off_img_1, off_img_1, exp_1 = select_offensive_inoffensive_image()
        not_off_img_2, off_img_2, exp_2 = select_offensive_inoffensive_image()

        not_off_img_1 = Image.open(f"/kaggle/input/facebook-hateful-memes/hateful_memes/{not_off_img_1}")
        off_img_1 = Image.open(f"/kaggle/input/facebook-hateful-memes/hateful_memes/{off_img_1}")
        not_off_img_2 = Image.open(f"/kaggle/input/facebook-hateful-memes/hateful_memes/{not_off_img_2}")
        off_img_2 = Image.open(f"/kaggle/input/facebook-hateful-memes/hateful_memes/{off_img_2}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": "Is this meme offensive? Anwer briefly. Give 1 line explanation only if it is offensive"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"This meme is offensive. {exp_1}"},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Inoffensive Image prompt (third image in the row)
                    {"type": "text", "text": "Is this image offensive? Answer briefly. Give 1 line explanation only if it is offensive"},
                ],
            },
            {
                "role": "assistant",
                "content": [                    
                    {"type": "text", "text":  "This meme is not offensive."},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt = True)
        inputs = processor(text=prompt, images=[not_off_img_1, off_img_1, not_off_img_2, off_img_2, Image.open(os.path.join(dir, df_test["img_path"][i]))], padding = True, return_tensors = "pt").to(model.device)

        # Generate the output for the target image
        generate_ids = model.generate(**inputs, max_new_tokens=30, temperature = 0.2, top_k = 5, top_p = 0.9, do_sample = True)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        img_path = os.path.join(dir, df_test["img_path"][i])
        # Append the result to the results list
        results_2.append({"image_path": img_path, "output": output})
    print("done")
    return results_2

# %%
def main(seed = 42):
    seed_all(seed)
    """Pass the dataset choice and the prompting setting as input arguments"""

    parser = argparse.ArgumentParser(description="Choose the prompt setting and the dataset")
    parser.add_argument("--dataset", choices=["esnli", "meme"], default="meme")
    parser.add_argument("--shot_type", choices=["0_shot", "few_shot"], default="0_shot")

    args = parser.parse_args()
    dataset = args.dataset
    shot_type = args.shot_type

    if dataset == "esnli":
        IMAGE_DIR = "/kaggle/input/esnli-ve/imgs/imgs"
        df_test = pd.read_csv("/kaggle/input/esnli-ve/test_processed.csv")
        print(f"Running E-SNLI-VE evaluation on {len(df_test)} samples...")

        if shot_type == "0_shot":        
            results = zero_shot_esnli_classification(df_test, IMAGE_DIR, seed=42)
        # save_results(results, df_test, "llava_0shot_snliVE.csv", "snli_ve")

        elif shot_type == "few_shot":
            results = few_shot_esnli_classification(df_test, IMAGE_DIR, seed=42)

        else:
            print("Invalid dataset. Choose '0_shot' or 'few_shot'")

        results_df = pd.DataFrame({
            "img_path": df_test["Flickr30kID"].tolist(),
            "hypothesis": df_test["hypothesis"].tolist(),
            "explanation": results
        })


    elif dataset == "meme":
        IMAGE_DIR = "/kaggle/input/facebook-hateful-memes/hateful_memes"
        df_test = pd.read_csv("/kaggle/input/train-prompt/train_prompt (1).csv")  # Update path as needed        
        print(f"Running meme classification on {len(df_test)} samples...")

        if shot_type == "0_shot":
            results = zero_shot_meme_classification(df_test, IMAGE_DIR, seed=42)
        # save_results(results, df_test, "llava_0shot_meme.csv", "meme")

        elif shot_type == "few_shot":
            results = few_shot_meme_classification(df_test, IMAGE_DIR, seed=42)

        else:
            print("Invalid dataset. Choose '0_shot' or 'few_shot'")

        results_df = pd.DataFrame({
            "img_path": df_test["img_path"].tolist(),
            "explanation": results
        })

    else:
        print("Invalid dataset. Choose 'esnli' or 'meme'")
    
    results_df.to_csv("answer_df.csv", index=False)
    print(f"Results saved to answer_df.csv")

if __name__ == "__main__":
    main()


