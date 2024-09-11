# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：Dr Zhu, 微信: 15000361164
@version：
@Date：
@Desc: run training for a qwen lora ft
"""

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed, DataCollatorForSeq2Seq, GenerationConfig, get_scheduler, AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

sys.path.append("./")
from calc_gen_scores import calc_scores
from io_utils import load_jsonl

from tokenization_qwen2 import Qwen2Tokenizer
from configuration_qwen2 import Qwen2Config
from modeling_qwen2 import Qwen2ForCausalLM

os.environ["WANDB_MODE"] = "disabled"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "where to store the cached data."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    num_train_epochs : Optional[int] = field(default=2)

    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.3)
    lora_alpha : Optional[float] = field(default=32.)
    adapter_rank: Optional[int] = field(default=8)
    adapter_dropout: Optional[float] = field(default=0.2)

    modules_to_save : Optional[str] = field(default='embed_tokens,lm_head')
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    leraning_rate : Optional[float] = field(default=1e-5)

    predict_with_generate : Optional[bool] = field(default=False)
    do_generation : Optional[bool] = field(default=False)

    tunable_param_names: Optional[str] = field(
        default=None,
        metadata={"help": "separate by comma; keywords for filtering tunable adapter/lora params"},
    )

    do_train: Optional[bool] = field(default=False)
    do_eval: Optional[bool] = field(default=False)
    do_retrain: Optional[bool] = field(default=False)
    gradient_align: Optional[bool] = field(
        default=False,
        metadata={"help": "use gradient alignment."},
    )
    # alignment_mode
    alignment_mode: Optional[str] = field(default="soft")

    eval_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = field(default=5e-5)

    # weights for regularizing loss terms
    orth_loss_weight: Optional[float] = field(default=0.5)
    sparse_loss_weight: Optional[float] = field(default=0.5)
    # prune_steps: Optional[int] = field(default=400)
    ratios_to_drop: Optional[int] = field(default=400)

    # search_space : Optional[str] = field(default="micro")
    #
    #
    # # training_args.start_search_steps and completed_steps % training_args.search_every == 0 and completed_steps <= training_args.end_search_steps
    start_prune_steps: Optional[int] = field(default=10000000)
    prune_every: Optional[int] = field(default=200)
    end_prune_steps: Optional[int] = field(default=1200)

    max_patience: Optional[int] = field(default=10)
    lora_rank2score_dir: Optional[str] = field(default=None)
    groups_number_sapce: Optional[int] = field(default=8)

    # 在评估每个LoRA模块时，是否对其他层的lora模块进行部分剪枝
    prune_others_for_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "在评估每个LoRA模块时，是否对其他层的lora模块进行部分剪枝."},
    )


logger = logging.getLogger(__name__)


def eval_model(model, eval_dataloader,):
    model.eval()
    losses = []
    total_loss = 0.0
    num_batches = 0
    for step, batch in tqdm(enumerate(eval_dataloader)):
        # batch["layer_attn_gates"] = layer_attn_gates
        # batch["layer_ffn_gates"] = layer_ffn_gates
        with torch.no_grad():
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.to(torch.float32).cpu().numpy().tolist()
            num_batches += 1

    try:
        eval_loss = total_loss / num_batches
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
        eval_loss = 1000000000

    return eval_loss


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = Qwen2Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.lora_rank = training_args.lora_rank
    config.lora_dropout = training_args.lora_dropout
    config.adapter_rank = training_args.adapter_rank
    config.adapter_dropout = training_args.adapter_dropout
    config.orth_loss_weight = training_args.orth_loss_weight
    config.sparse_loss_weight = training_args.sparse_loss_weight
    config.groups_number_sapce = training_args.groups_number_sapce
    config.prune_others_for_eval = training_args.prune_others_for_eval
    config._attn_implementation = "eager"

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = tokenizer.im_end_id

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def tokenize_function(example):

        # max_seq_length = 2048 + 512
        max_seq_length = block_size

        input = example["input"]
        target = example["target"]
        history = [(input, target)]

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role
            ) + nl_tokens + tokenizer.encode(content)

        def _tokenize_str_target(target):
            return (
                f"assistant\n",
                tokenizer.encode("assistant") + nl_tokens,
                target,
                tokenizer.encode(target)
            )

        system = "You are a helpful assistant."

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        input_ids = []
        labels = []
        instruction = ""

        input_ids.extend(system_tokens)
        labels.extend([-100] * len(system_tokens))
        instruction += f"{im_start}{system_text}{im_end}\n"

        for turn_query, turn_response in history:
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            assistant_text, assistant_tokens_part = _tokenize_str("assistant", turn_response)
            assistant_tokens = im_start_tokens + assistant_tokens_part + im_end_tokens

            instruction = instruction + f"{im_start}{query_text}{im_end}\n{im_start}{assistant_text}{im_end}"
            input_ids = input_ids + \
                        nl_tokens + im_start_tokens + query_tokens_part + im_end_tokens + \
                        nl_tokens + im_start_tokens + assistant_tokens_part + im_end_tokens

            labels = labels + \
                     [-100] * len(nl_tokens + im_start_tokens + query_tokens_part + im_end_tokens) + \
                     [-100] * len(nl_tokens + im_start_tokens) + assistant_tokens_part + im_end_tokens
            # labels = copy.copy(input_ids)

        # print("input_ids: ", input_ids)
        # print("labels: ", labels)

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]

        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(labels) == len(attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # "length": original_length,
        }

    def tokenize_function_eval(example):

        # max_seq_length = 1024

        input = example["input"]
        # target = example["target"]

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role
            ) + nl_tokens + tokenizer.encode(content)

        def _tokenize_str_target(target):
            return (
                f"assistant\n",
                tokenizer.encode("assistant") + nl_tokens,
                target,
                tokenizer.encode(target)
            )

        system = "You are a helpful assistant."

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        query_text, query_tokens_part = _tokenize_str("user", input)
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
        resp_text_0, resp_token_0, _, _ = _tokenize_str_target(
            ""
        )
        resp_token_0 = im_start_tokens + resp_token_0
        # resp_token_1 = resp_token_1 + im_end_tokens

        instruction = f"{im_start}{system_text}{im_end}\n{im_start}{query_text}{im_end}\n{im_start}{resp_text_0}"
        # response = f"{resp_text_1}{im_end}"
        # raw_text = instruction + response
        # print("raw_text: ", raw_text)

        input_ids = system_tokens + nl_tokens + query_tokens + nl_tokens + resp_token_0
        # labels = [-100] * len(system_tokens + nl_tokens + query_tokens + nl_tokens + resp_token_0) + resp_token_1

        # if len(input_ids) > max_seq_length:
        #     input_ids = input_ids[-max_seq_length:]
        #     labels = labels[-max_seq_length:]

        attention_mask = [1] * len(input_ids)
        # if len(input_ids) < max_seq_length:
        #     input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
        #     labels = labels + [-100] * (max_seq_length - len(labels))
        #     attention_mask = attention_mask + [0] * (max_seq_length - len(attention_mask))

        # print("raw_text: ", raw_text)

        # assert len(input_ids) == len(labels) == len(attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "labels": labels,
            # "length": original_length,
        }


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size + 1) * block_size
        # Split by chunks of max_len.
        result = {}
        for k, t in concatenated_examples.items():
            if total_length > len(t):
                if k == "input_ids":
                    t = t + [tokenizer.eos_token_id] * (total_length - len(t))
                elif k == "attention_mask":
                    t = t + [0] * (total_length - len(t))
                else:
                    t = t + [-100] * (total_length - len(t))

            truncs = [t[i : i + block_size] for i in range(0, total_length, block_size)]
            result[k] = truncs

        # for k in result:
        #     print(k, len(result[k]))
        return result

    # with training_args.main_process_first(desc="dataset map tokenization and grouping"):
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.dataset_name, "train.json"),
            "dev": os.path.join(data_args.dataset_name, "dev.json"),
            # "test": os.path.join(data_args.dataset_name, "test.json"),
        },
        # cache_dir=data_args.dataset_cache_dir,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    os.makedirs(os.path.join(training_args.output_dir, f'cache'), exist_ok=True)
    tokenized_dataset = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=8,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=True,
                cache_file_names={k: os.path.join(training_args.output_dir, f'cache/tokenized_{k}.arrow') for k in raw_datasets},
                desc="Running tokenizer on dataset",
            )
    print("tokenized_dataset: ", tokenized_dataset)
    print(tokenized_dataset["train"][3]['input_ids'])
    print(tokenized_dataset["train"][3]['labels'])

    # tokenized_dataset = tokenized_dataset.map(
    #     group_texts,
    #     batched=True,
    #     # batch_size=1024,
    #     num_proc=8,
    #     load_from_cache_file=True,
    #     keep_in_memory=False,
    #     cache_file_names = {k: os.path.join(data_args.dataset_name, f'cache/grouped_{k}.arrow') for k in tokenized_dataset},
    #     desc=f"Grouping texts in chunks of {block_size}",
    # )

    lm_datasets = tokenized_dataset

    # lm_datasets = tokenized_dataset["train"].train_test_split(test_size=0.02)
    # lm_datasets["dev"] = lm_datasets["test"]
    print(lm_datasets)

    test_dataset = raw_datasets["dev"].map(
        tokenize_function_eval,
        batched=False,
        num_proc=1,
        desc="Running tokenizer on test dataset",
    )
    print(test_dataset)

    print(tokenizer.decode(test_dataset[1]['input_ids']))
    print(tokenizer.decode(test_dataset[3]['input_ids']))
    print(tokenizer.decode(test_dataset[10]['input_ids']))

    # if training_args.do_train:
    train_dataset = lm_datasets['train']
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    logger.info(f"Num train_samples  {len(train_dataset)}")
    logger.info("training example:")
    logger.info(tokenizer.decode(train_dataset[0]['input_ids']))

    # if training_args.do_eval:
    eval_dataset = lm_datasets["dev"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    logger.info(f"Num eval_samples  {len(eval_dataset)}")
    logger.info("training example:")
    logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    # torch_dtype = model_args.torch_dtype
    # config.num_hidden_layers = 2
    # model = QWenLMHeadModel._from_config(
    #     config
    # )
    model = Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.bfloat16,
    )

    # init for lora params
    for idx in range(len(model.model.list_lora_a_params)):
        torch.nn.init.uniform_(model.model.list_lora_a_params[idx].weight, -0.1, 0.1)
        torch.nn.init.zeros_(model.model.list_lora_b_params[idx].weight)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding='longest'
    )

    # Initialize our Trainer
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    eval_indices = list(range(len(eval_dataset)))
    random.shuffle(eval_indices)
    valid_dataset = eval_dataset.select(eval_indices[: 196])
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size * 2
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    tunable = training_args.tunable_param_names.strip().split(",")
    print([n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and any(nd in n for nd in tunable) ])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and any(nd in n for nd in tunable) ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and any(nd in n for nd in tunable) ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_train_steps * training_args.gradient_accumulation_steps,
    )

    # accelerator
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["project_dir"] = training_args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        **accelerator_log_kwargs
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        os.makedirs(training_args.output_dir, exist_ok=True)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    output_dir_stage_1 = training_args.output_dir
    os.makedirs(output_dir_stage_1, exist_ok=True)
    if training_args.do_train:

        # Train!
        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        steps = 0
        starting_epoch = 0

        # Training stage 1:
        # 第一阶段：lora参数初步训练
        total_model_params = 0
        num_trained_params = 0
        for n, p in model.named_parameters():
            # if ("lora_vector" in n):
            # if ("lora_vector" in n) or ("lora_a" in n) or ("lora_b" in n) or ("adapter" in n):
            # if ("lora_a" in n) or ("lora_b" in n) or ("adapter" in n):
            if any(nd in n for nd in tunable) :
                p.requires_grad = True
            else:
                p.requires_grad = False
            if p.requires_grad:
                num_trained_params += p.numel()
            else:
                total_model_params += p.numel()
            print(n, p.requires_grad, p.numel())

        logger.info("Total Model Parameters: {}, "
                    "Trainable Parameters: {}".format(
            total_model_params, num_trained_params))

        # training loop
        best_loss = 1000000000000
        best_loss_full_model = 1000000000000
        best_steps = None
        best_steps_full_model = None
        max_patience = training_args.max_patience
        patience = 0

        # 用于记录已经mask的lora ranks
        list_ranks_to_mask = []

        if training_args.peft_path:
            model.model.list_lora_a_params = torch.load(
                os.path.join(training_args.peft_path, "list_lora_a_params.bin")
            )
            model.model.list_lora_b_params = torch.load(
                os.path.join(training_args.peft_path, "list_lora_b_params.bin")
            )

        eval_loss = eval_model(
            model,
            eval_dataloader,
        )
        logger.info(f"completed_steps: {completed_steps}; eval loss: {eval_loss}")

        for epoch in range(starting_epoch, training_args.num_train_epochs):

            total_loss = 0
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                steps += 1
                model.train()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                if random.uniform(0, 1) < 0.1:
                    print("loss: ", loss)

                if random.uniform(0, 1) < 0.01:

                    print("model.model.list_lora_a_params[2].weight.data[: 1]: ",
                          model.model.list_lora_a_params[2].weight.data[: 1])
                    print("model.model.list_lora_b_params[2].weight.data[: 1]: ",
                          model.model.list_lora_b_params[2].weight.data[: 1])

                if steps % training_args.gradient_accumulation_steps == 0:
                    # print("steps: ", steps)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_args.max_grad_norm
                    )

                    completed_steps += 1
                    print("completed_steps: ", completed_steps)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    if completed_steps % training_args.eval_steps == 0 or completed_steps == 25 or completed_steps == 5:

                        # eval model with structural drop
                        eval_loss = eval_model(
                            model,
                            eval_dataloader,
                        )
                        logger.info(f"completed_steps: {completed_steps}; eval loss: {eval_loss}")
                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_steps = completed_steps
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)

                            torch.save(
                                unwrapped_model.model.list_lora_a_params,
                                os.path.join(output_dir_stage_1, "list_lora_a_params.bin")
                            )
                            torch.save(
                                unwrapped_model.model.list_lora_b_params,
                                os.path.join(output_dir_stage_1, "list_lora_b_params.bin")
                            )

                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(output_dir_stage_1)

                            patience = 0

                        else:
                            patience += 1

                            # logger.info(f"best_loss: {best_loss}; best_steps: {best_steps}")
                        logger.info(f"current best_loss: {best_loss}; best_steps: {best_steps}")

                        if patience >= max_patience:
                            break

            if completed_steps >= training_args.max_train_steps:
                break
            if patience >= max_patience:
                break

        logger.info("*" * 50)
        logger.info(f"best steps: {best_steps}; best loss: {best_loss}")
        logger.info("*" * 50)

    list_ranks_to_mask = []
    if training_args.do_eval:
        ###############################
        # 进行lora的重要性评估
        ###############################

        model.eval()

        # 加载lora参数
        if training_args.peft_path:
            model.model.list_lora_a_params = torch.load(
                os.path.join(output_dir_stage_1, "list_lora_a_params.bin")
            )
            model.model.list_lora_b_params = torch.load(
                os.path.join(output_dir_stage_1, "list_lora_b_params.bin")
            )
        model.eval()

        # 先跑一遍验证集
        # eval_loss = eval_model(
        #     model,
        #     eval_dataloader,
        # )
        # logger.info(f"eval loss: {eval_loss}")

    if training_args.do_generation:

        model.model.list_lora_a_params = torch.load(
            os.path.join(training_args.output_dir, "list_lora_a_params.bin")
        )
        model.model.list_lora_b_params = torch.load(
            os.path.join(training_args.output_dir, "list_lora_b_params.bin")
        )
        model.eval()

        generation_config = GenerationConfig.from_dict(
            {
                "chat_format": "chatml",
                "eos_token_id": 151643,
                "pad_token_id": 151643,
                "max_window_size": 6144,
                "max_new_tokens": 1024,
                "do_sample": False,
                "top_k": 0,
                "top_p": 0.0,
                "num_beams": 5,
                "repetition_penalty": 1.05
            }
        )
        model.generation_config = generation_config
        print("model.generation_config: ", model.generation_config)

        # 加载数据
        test_samples_origin = load_jsonl(os.path.join(data_args.dataset_name, "dev.json"))
        test_samples_gen = []
        for test_samp in test_samples_origin:
            history_1 = [(test_samp["input"], test_samp["target"])]

            history = history_1[: -1]
            query = history_1[-1][0]
            targeted_response = history_1[-1][1]

            tokenized_samp = tokenize_function_eval(
                {"input": query}
            )
            input_ids = [tokenized_samp["input_ids"]]
            attention_mask = [tokenized_samp["attention_mask"]]
            input_length = len(input_ids[0])

            outputs = model.generate(
                torch.LongTensor(input_ids).to(torch.device("cuda:0")),
                attention_mask=torch.LongTensor(attention_mask).to(torch.device("cuda:0")),
                generation_config=generation_config,
            )
            response = outputs[0][input_length:]
            print("response: ", response)

            eod_token_idx = None
            for i in range(len(response)):
                if response[i] in [151645, tokenizer.eos_token_id]:
                    eod_token_idx = i
                    break
            if eod_token_idx is None:
                eod_token_idx = len(response) - 1

            response = response[: eod_token_idx]
            response_text = tokenizer.decode(
                response
            )
            print("response_text: ", response_text)
            samp_copy = copy.deepcopy(test_samp)
            samp_copy["pred"] = response_text
            samp_copy["target"] = history_1[0][1]
            test_samples_gen.append(
                samp_copy
            )

            with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
                for samp in test_samples_gen:
                    f.write(
                        json.dumps(samp, ensure_ascii=False) + "\n"
                    )

        scores = calc_scores(os.path.join(training_args.output_dir, "test_predictions.json"))
        print("*" * 50)
        print("scores: ", scores)
        print("*" * 50)


if __name__ == "__main__":
    main()

