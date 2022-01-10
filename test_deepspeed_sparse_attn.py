import torch
from torch import nn
from deepspeed.ops.sparse_attention import SparseSelfAttention
from deepspeed.ops.sparse_attention.sparsity_config import BigBirdSparsityConfig
import time

batch_size = 32
num_attention_heads = 1
size_per_head = 512
num_rand_blocks = 3
from_seq_length = 4096
to_seq_length = 4096
from_block_size = 64
to_block_size = 64

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)

# only support training in fp16 currently
query_layer = torch.rand(
    batch_size, num_attention_heads, from_seq_length, size_per_head
).half().cuda()
key_layer = torch.rand(
    batch_size, num_attention_heads, to_seq_length, size_per_head
).half().cuda()
value_layer = torch.rand(
    batch_size, num_attention_heads, to_seq_length, size_per_head
).half().cuda()

start = time.perf_counter()
sparse_config = BigBirdSparsityConfig(
    num_heads=num_attention_heads,
    block=from_block_size,
    different_layout_per_head=True,
    num_random_blocks=num_rand_blocks,
    num_sliding_window_blocks=3,  # use default value in this config class
    num_global_blocks=1,  # also the FIXED window/global size in 'pytorch_bigbird' implementation
)

attention_layer = SparseSelfAttention(
    sparsity_config=sparse_config, max_seq_length=4096
).cuda()

attn_output = attention_layer(query_layer, key_layer, value_layer)


end = time.perf_counter()
print(batch_size*num_attention_heads*from_seq_length/(end - start))
