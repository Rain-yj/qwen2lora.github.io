import torch
from modelscope import snapshot_download

# model_dir = snapshot_download(
#     "modelscope/Llama-2-7b-chat-ms", revision='v1.0.5',
#     ignore_file_pattern=[r'.+\.bin$'],
#     # ignore_file_pattern=[r'*bin'],
#     cache_dir="./resources/Llama-2-7b-chat-hf"
# )


model_dir = snapshot_download(
    # "AI-ModelScope/stable-diffusion-v1-4", revision='v1.0.2',
    # "qwen/Qwen1.5-7B-Chat",
    # "qwen/Qwen1.5-7B",
    # "qwen/Qwen1.5-0.5B",
    "qwen/Qwen1.5-0.5B-Chat",
    # revision='v1.0.2',
    # ignore_file_pattern=[r'.+\.bin$'],
    # ignore_file_pattern=[r'*bin'],
    cache_dir="./resources/"
)