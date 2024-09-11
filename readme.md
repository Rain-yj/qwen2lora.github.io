


## 下载 模型

```bash

python model_download.py

````








## belle

```bash

# LoRA参数训练
nohup python -u src/qwen/run_sft.py --seed 100 --dataset_name datasets/belle --model_name_or_path ./resources/qwen/Qwen1___5-7B --block_size 1024 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --num_train_epochs 10 --warmup_steps 200 --output_dir experiments/outputs/qwen_7b_belle_100 --do_train --eval_steps 50 --learning_rate 2e-4 --tunable_param_names lora_a,lora_b --overwrite_output_dir --max_patience 10 --loss_type entropy --lora_rank 64 --lora_modules_to_keep 1 > experiments/logs/qwen_7b_belle_100.log &



```




## qwen1.5-0.5B-chat


```bash

python -u run_sft.py --seed 100 --dataset_name datasets/alpaca --model_name_or_path ./resources/qwen/Qwen1___5-0___5B-Chat --block_size 1024 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --num_train_epochs 10 --warmup_steps 200 --output_dir experiments/outputs/qwen_0.5b_alpaca_100 --do_train --do_generation --eval_steps 50 --learning_rate 2e-4 --tunable_param_names lora_a,lora_b --overwrite_output_dir --max_patience 10 --lora_rank 32


python -u run_sft.py --seed 100 --dataset_name datasets/alpaca --model_name_or_path ./resources/qwen/Qwen1___5-0___5B-Chat --block_size 1024 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --num_train_epochs 10 --warmup_steps 200 --output_dir experiments/outputs/qwen_0.5b_alpaca_100 --do_generation --eval_steps 50 --learning_rate 2e-4 --tunable_param_names lora_a,lora_b --overwrite_output_dir --max_patience 10 --lora_rank 32



```