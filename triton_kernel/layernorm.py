import torch
import triton
import triton.language as tl
import torch.nn as nn

LN_FWD_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE": 128},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 256},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 512},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 1024},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 512},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 1024},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 2048},
        num_warps=8,
        num_stages=2,
    ),
]
LN_BWD_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE": 256},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE": 512},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE": 1024},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE": 512},
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE": 1024},
        num_warps=8,
    ),
]
LN_DWDB_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256},
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256},
        num_warps=8,
    ),
]
eps = 1e-5

@triton.autotune(
    configs=LN_FWD_CONFIGS,
    key=["N"],
    cache_results=True,
)
@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    x_ptr += pid * N
    y_ptr += pid * N

    sum_for_mean = 0.0
    sum_for_std = 0.0

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        sum_for_mean += tl.sum(x, 0) # type: ignore

    mean = sum_for_mean / N

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        sum_for_std += tl.sum(diff * diff, 0) # type: ignore

    var = sum_for_std / N
    rstd = tl.rsqrt(var + 1e-5)

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask, other=0.0)
        y = weight * (x - mean) * rstd + bias
        tl.store(y_ptr + offsets, y, mask)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)

@triton.autotune(
    configs=LN_BWD_CONFIGS,
    key=["N"],
    reset_to_zero=[
        "dw_intermediate_ptr",
        "db_intermediate_ptr",
        "key_ptr",
    ],
    cache_results=True,
)
@triton.jit
def _layernorm_backward_dx(
    x_ptr, dy_ptr, dx_ptr,
    weight_ptr, mean_ptr, rstd_ptr,
    dw_intermediate_ptr, db_intermediate_ptr, key_ptr,
    N, MEM_SZIE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    x_ptr += pid * N
    dx_ptr += pid * N
    dy_ptr += pid * N
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    accumulator_dx_hat = 0.0
    accumulator_dot = 0.0

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dx_hat = dy * weight
        accumulator_dx_hat += tl.sum(dx_hat, 0) # type: ignore
        accumulator_dot += tl.sum(x_hat * dx_hat, 0) # type: ignore

    arg1 = accumulator_dx_hat / N
    arg2 = accumulator_dot / N

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dx_hat = dy * weight

        tl.store(dx_ptr + offsets, rstd * (dx_hat - arg1 - x_hat * arg2), mask)

    

    key_id = pid % MEM_SZIE
    dw_intermediate_ptr += key_id * N
    db_intermediate_ptr += key_id * N

    key_ptr += key_id
    count_ptr = key_ptr + MEM_SZIE

    while tl.atomic_cas(key_ptr, 0, 1) == 1:
        pass

    count = tl.load(count_ptr)
    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        partial_dw = (x - mean) * rstd * dy
        partial_db = dy
        
        if(count == 1):
            partial_dw += tl.load(dw_intermediate_ptr + offsets, mask, other=0.0)
            partial_db += tl.load(db_intermediate_ptr + offsets, mask, 0.0)

        tl.store(dw_intermediate_ptr + offsets, partial_dw, mask)
        tl.store(db_intermediate_ptr + offsets, partial_db, mask)

    tl.atomic_xchg(count_ptr, 1)

    tl.debug_barrier()

    tl.atomic_xchg(key_ptr, 0)

@triton.autotune(
    configs=LN_DWDB_CONFIGS,
    key=["N", "MEM_SIZE"],
    cache_results=True,
)
@triton.jit
def _layernorm_backward_dwdb(
    dw_intermediate_ptr, db_intermediate_ptr,
    dw_ptr, db_ptr,
    N, MEM_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols < N

    dw = tl.zeros((BLOCK_SIZE_N, ), tl.float32)
    db = tl.zeros((BLOCK_SIZE_N, ), tl.float32)

    for i in tl.range(0, MEM_SIZE, BLOCK_SIZE_M): # type: ignore
        row = i + tl.arange(0, BLOCK_SIZE_M)
        row_mask = row < MEM_SIZE
        offsets = row[:, None] * N + cols[None, :]
        mask = row_mask[:, None] & cols_mask[None, :]
        acc_w = tl.load(dw_intermediate_ptr + offsets, mask, 0.0)
        acc_b = tl.load(db_intermediate_ptr + offsets, mask, 0.0)
        dw += tl.sum(acc_w, 0)
        db += tl.sum(acc_b, 0)

    tl.store(dw_ptr + cols, dw, cols_mask)
    tl.store(db_ptr + cols, db, cols_mask)

class Layernorm_wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        x = x.contiguous()
        y = torch.empty_like(x)

        M, N = x.reshape(-1, x.shape[-1]).shape
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        _layernorm_forward[(M, )](
            x, y,
            weight, bias,
            mean, rstd,
            N
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(bias)

        MEM_SIZE = 256

        db_intermediate = torch.zeros((MEM_SIZE, N), device=x.device)
        dw_intermediate = torch.zeros((MEM_SIZE, N), device=x.device)

        key = torch.zeros((2 * MEM_SIZE, ),dtype=torch.int32, device=x.device)

        _layernorm_backward_dx[(M, )](
            x, dy, dx, weight, mean, rstd, dw_intermediate, db_intermediate,
            key, N,
            MEM_SIZE # type: ignore
        )

        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layernorm_backward_dwdb[grid](
            dw_intermediate, db_intermediate,
            dw, db,
            N, MEM_SIZE #type: ignore
        )

        return dx, dw, db

layernorm = Layernorm_wrapper.apply

class LayerNorm_triton(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(embed_size)
        )
        self.bias = nn.Parameter(
            torch.zeros(embed_size)
        )

    def forward(self, x):
        return layernorm(x, self.weight, self.bias)