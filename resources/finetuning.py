'''
Script to finetune a pre-trained model. 
Script is based on 02-ft-unsloth.ipynb
Run this script under tmux to avoid terminal closure

NOTE
Please compare the script with latest script of your desired model from here: https://github.com/unslothai/unsloth?tab=readme-ov-file
If any difference, update or discuss.
'''

#importing dependencies
from unsloth import FastLanguageModel
import torch
import wandb
import os
import dotenv

_ = dotenv.load_dotenv(".env")

PROJECT_NAME = "Finetune_Tech_Blogs"   #please change this
RUN_NAME = "run_1"          #please change this
MODEL_NAME = 'unsloth/mistral-7b-v0.2-bnb-4bit'
SAVE_MODEL_NAME = 'mistral-7b-v0.2-quickreel'

#Initializing WandB
wandb.login(key=os.getenv("e35a85fe1fece13bb07176b39d1e570f6ee20039"))
wandb.init(project=PROJECT_NAME, name = RUN_NAME)


#training configs
max_seq_length = 4096               #Choose any! We auto support RoPE Scaling internally!
dtype = None                        #None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True                 #Use 4bit quantization to reduce memory usage. Can be False.
model_name = MODEL_NAME             #"unsloth/llama-3-8b-bnb-4bit" #"unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
save_model_name = SAVE_MODEL_NAME   #'llama-3-8b-quickreel'


# 4bit pre quantized models we support for 4x faster downloading + no OOMs. 
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/mistral-7b-v0.2-bnb-4bit", ## New Mistral 32K base model
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit" # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name ="unsloth/llama-3-8b-bnb-4bit",  #Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

#load the train_aug.jsonl
import pandas as pd
from datasets import Dataset

df = pd.read_json("./resources/train.jsonl", lines=True)
dataset = Dataset.from_pandas(df)
#dataset = load_dataset("Danbnyn/Bloomberg_Financial_News", split = "train") - load directly from HF if your dataset is from there
dataset = dataset.map(formatting_prompts_func, batched = True,)

print("Showing formatted dataset")
print(dataset[0])


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = None, #60
        num_train_epochs=2, #change according to the training time
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 100, #change according to the number of steps. total_steps//5 is reasonable. total_steps are shown by unsloth when you start the training. 
        report_to = "wandb",
    ),
)


print("\n\n## Starting model training")
trainer_stats = trainer.train()


print("\n\n## Showing final memory and time stats")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#Save the model after training
print(f"Saving the finetuned model at {save_model_name}")
model.save_pretrained(save_model_name)   #Local saving

# push it to huggingface
