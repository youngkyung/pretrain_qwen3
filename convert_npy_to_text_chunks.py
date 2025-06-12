import argparse
import json
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

EOS_TOKEN = 2
MINIMUM_LEN = 100
SAVE_EVERY = 10000  # 문장 10000개마다 저장
TOKENIZER_PATH = "/nfsdata/languageAI/riwoo/tokenizer/kt_v3_1_4_0/tiktoken/kt_v3.tiktoken"

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

def load_tiktoken_tokenizer(file_path):
    from tiktoken.load import load_tiktoken_bpe
    with open(file_path, 'r') as f:
        tiktoken_json = json.load(f)

    import base64
    mergeable_ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in tiktoken_json["mergeable_ranks"].splitlines() if line)
    }

    return tiktoken.Encoding(
        name=os.path.basename(file_path),
        pat_str=tiktoken_json['pat_str'],
        mergeable_ranks=mergeable_ranks,
        special_tokens=tiktoken_json['special_tokens']
    )

def load_npy_tokens(npy_path, dtype=np.int32):
    return np.fromfile(npy_path, dtype=dtype)

def split_by_eos(tokens, eos_token=EOS_TOKEN):
    split_indices = np.where(tokens == eos_token)[0]
    prev = 0
    for idx in split_indices:
        yield tokens[prev:idx]
        prev = idx + 1
    if prev < len(tokens):
        yield tokens[prev:]

def convert_and_save(npy_path, output_base_path, output_format="jsonl"):
    tokenizer = load_tiktoken_tokenizer(TOKENIZER_PATH)
    tokens = load_npy_tokens(npy_path)

    chunk_iter = split_by_eos(tokens, eos_token=EOS_TOKEN)
    output_dir = os.path.dirname(output_base_path)
    os.makedirs(output_dir, exist_ok=True)

    buffer = []
    file_count = 0
    total_saved = 0

    log(f"🚀 시작: {npy_path}")

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing")):
        if len(chunk) < MINIMUM_LEN:
            continue

        try:
            text = tokenizer.decode(chunk.tolist()).strip()
            if not text:
                continue
            buffer.append({"text": text})
        except Exception as e:
            continue

        # SAVE_EVERY 개마다 저장
        if len(buffer) >= SAVE_EVERY:
            part_path = f"{output_base_path}_part{file_count:03d}.{output_format}"
            with open(part_path, "w", encoding="utf-8") as fout:
                for item in buffer:
                    if output_format == "jsonl":
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:
                        fout.write(item["text"] + "\n")
            log(f"💾 저장됨: {part_path} | 문장 수: {len(buffer)}")

            total_saved += len(buffer)
            file_count += 1
            buffer = []

    # 마지막 남은 데이터 저장
    if buffer:
        part_path = f"{output_base_path}_part{file_count:03d}.{output_format}"
        with open(part_path, "w", encoding="utf-8") as fout:
            for item in buffer:
                if output_format == "jsonl":
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    fout.write(item["text"] + "\n")
        log(f"✅ 마지막 저장됨: {part_path} | 문장 수: {len(buffer)}")
        total_saved += len(buffer)

    log(f"🎉 전체 완료 | 총 저장 문장 수: {total_saved}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--output_base_path", type=str, required=True, help="Output file path without extension")
    parser.add_argument("--format", type=str, choices=["jsonl", "txt"], default="jsonl", help="Output format")
    args = parser.parse_args()

    convert_and_save(args.npy_path, args.output_base_path, args.format)
