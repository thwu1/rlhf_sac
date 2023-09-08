import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SACConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.trainer.accelerate_sac_trainer import AccelerateSACTrainer
from trlx.utils.modeling import (
    freeze_bottom_causal_layers,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        eval_batch_size=32,
        max_history_size=128,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateSACTrainer",
        checkpoint_dir="checkpoints/sac_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=SACConfig(
        name="SACConfig",
        num_rollouts=64,
        # chunk_size=16,
        chunk_size=4,
        sac_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        alpha=1,
        # beta=1,
        gamma=0.001,
        lam=0.95,
        num_new_actions=1,
        actor_coef=1,
        adv_scale=1,
        # actor_reg_coef=0.9,
        # cliprange=0.2,
        # cliprange_value=0.2,
        # vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


default_config.train.batch_size = 8
default_config.train.total_steps = 1500
default_config.train.checkpoint_dir = "checkpoints/sac_hh_125M"
default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
default_config.method.num_rollouts = 128
default_config.train.tracker = None

config = TRLConfig.update(default_config, {})

acceleratesactrainer = AccelerateSACTrainer(config)
model = acceleratesactrainer.get_arch(config)
freeze_bottom_causal_layers(model.base_model, config.model.num_layers_unfrozen)

tokenizer = acceleratesactrainer.tokenizer
prompt = "Human: How can I stop myself from experiencing road rage?\n\nAssistant:"
prompt_tensor = tokenizer(prompt, return_tensors="pt")["input_ids"]
mask = torch.ones_like(prompt_tensor)
# print(prompt)
# tokens = torch.load("tokens.pt")
# print(tokens)

samples = acceleratesactrainer.generate(prompt_tensor, mask)
samples1 = acceleratesactrainer.generate(prompt_tensor, mask)
print(samples.shape)
print(tokenizer.decode(samples[0]))
print("\n")
print(tokenizer.decode(samples1[0]))

# attention_mask = tokens.not_equal(tokenizer.pad_token_id).long()
# outputs = model(tokens, attention_mask=attention_mask, return_dict=True)
# print("\nOutput:", tokenizer.decode(tokens[0]))

# logits = outputs.logits
# logprob = torch.nn.functional.log_softmax(logits, dim=-1)

# old_qa = torch.gather(logits[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1).detach()

# print(old_qa.dtype)
# print(torch.gather(logprob[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1))

# trlx.train(
#     prompts=prompts,
#     eval_prompts=eval_prompts,
#     reward_fn=reward_fn,
#     config=config,
#     stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
# )
