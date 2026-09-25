import torch
import torch.nn.functional as F

import tiktoken

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.triton_model import triton_GPT, GPTConfig
from src.dataloader import Dataloader

from baseline import get_args, measure

def one_run(model, optimizer, dataloader, device):

    optimizer.zero_grad(set_to_none=True)
    x, labels = dataloader.next_batch()
    x, labels = x.to(device), labels.to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.view(-1)
            )

    loss.backward()
    optimizer.step()

def main():
    
    args = get_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    file_path = Path(__file__).resolve().parent / "data" / "input.txt"
    with open(file_path, 'r') as f:
        text = f.read()

    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(text)

    batch_size = 16
    window_size = 512

    lr = 1e-4
    embed_size = 512
    head_nums = 8
    block_nums = 8
    vocab_size = 50304

    device = "cuda"

    torch.set_float32_matmul_precision('high')

    config = GPTConfig(
        token_nums=vocab_size,
        embed_size=embed_size,
        head_nums=head_nums,
        window_size=window_size,
        block_nums=block_nums,
    )

    model = triton_GPT(config)
    model.to(device)
    if args.compile:
        model = torch.compile(model)
        print("use compile model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = Dataloader(tokens, batch_size, window_size)

    avg_time, peak_mem = measure(
        lambda : one_run(
            model, optimizer, dataloader, device
            )
    )

    tokens_per_step = batch_size * window_size
    tokens_per_second = tokens_per_step / avg_time
    peak_memory_gb = peak_mem / 1024**3

    print(f"Tokens per step:   {tokens_per_step:,}")
    print(f"Step time:         {avg_time * 1000:.3f} ms")
    print(f"Tokens/s:          {tokens_per_second:,.0f}")
    print(f"Peak GPU memory:   {peak_memory_gb:.3f} GB")

if __name__ == "__main__":
    main()