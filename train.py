import os
import json
import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

# ======== CONFIG ========
model_dir = "/home/ygseo/cross_lingual/Qwen3-0.6B"
data_jsonl = "/home/ygseo/cross_lingual/my_corpus.jsonl"
data_npy = "/home/ygseo/cross_lingual/my_corpus.npy"
output_dir = "/home/ygseo/cross_lingual/qwen3_manual_pretrain_ckpt"
block_size = 1024
batch_size = 1
epochs = 1
lr = 2e-5
save_every = 10  # steps
use_fp16 = torch.cuda.is_available()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# =========================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def load_npy_dataset(npy_file, block_size=1024, token_dtype=np.int32):
    try:
        with open(npy_file, 'rb') as f:
            raw = f.read()

        num_tokens = len(raw) // np.dtype(token_dtype).itemsize
        tokens = np.frombuffer(raw, dtype=token_dtype, count=num_tokens)

        # Split into blocks
        num_blocks = len(tokens) // block_size
        trimmed_tokens = tokens[:num_blocks * block_size]
        blocks = trimmed_tokens.reshape((num_blocks, block_size))

        # Wrap as list of dicts
        data = [{"input_ids": block.tolist()} for block in blocks]

        return Dataset.from_list(data)

    except Exception as e:
        print(f"❌ Error loading binary NPY file: {e}")
        raise

# ========== DDP INIT ==========
setup_ddp()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
rank = dist.get_rank()
world_size = dist.get_world_size()

# ========== LOAD MODEL ==========
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model = model.to(device)
model = DDP(model, device_ids=[device])

# ========== DATA LOAD ==========
def load_jsonl_dataset(path):
    return load_dataset("json", data_files=path, split="train")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=block_size)

jsonl_data = load_jsonl_dataset(data_jsonl)
jsonl_tokenized = jsonl_data.map(tokenize_fn, remove_columns=["text"])

npy_data = load_npy_dataset(data_npy, block_size=block_size)

dataset = concatenate_datasets([jsonl_tokenized, npy_data])
sampler = DistributedSampler(dataset)

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone()
    }

dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

# ========== OPTIMIZER & SCALER ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler(enabled=use_fp16)
start_step = 0

# ========== TRAIN LOOP ==========
model.train()
global_step = start_step
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    loop = tqdm(dataloader, desc=f"[RANK {rank}] Epoch {epoch+1}/{epochs}", disable=rank != 0)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=use_fp16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if rank == 0:
            loop.set_postfix(loss=loss.item(), step=global_step)

            if global_step % save_every == 0:
                os.makedirs(output_dir, exist_ok=True)
                ckpt_name = f"ckpt_{timestamp}_step{global_step}.pt"
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": global_step,
                    "loss": loss.item()
                }, os.path.join(output_dir, ckpt_name))
                print(f"💾 [RANK 0] Saved checkpoint: {ckpt_name}")

        global_step += 1

    if rank == 0:
        print(f"✅ [RANK 0] Finished Epoch {epoch+1}")

cleanup_ddp()
