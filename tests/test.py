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

# default_config = TRLConfig(
#     train=TrainConfig(
#         seq_length=1024,
#         epochs=10000,
#         total_steps=10000,
#         batch_size=4,
#         checkpoint_interval=10000,
#         eval_interval=500,
#         pipeline="PromptPipeline",
#         trainer="AccelerateSACTrainer",
#         checkpoint_dir="checkpoints/sac_hh",
#     ),
#     model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
#     tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
#     optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
#     scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
#     method=SACConfig(
#         name="SACConfig",
#         num_rollouts=64,
#         # chunk_size=16,
#         chunk_size=4,
#         sac_epochs=4,
#         init_kl_coef=0.05,
#         target=6,
#         horizon=10000,
#         gamma=1,
#         lam=0.95,
#         cliprange=0.2,
#         cliprange_value=0.2,
#         vf_coef=1,
#         scale_reward="running",
#         ref_mean=None,
#         ref_std=None,
#         cliprange_reward=10,
#         gen_kwargs=dict(
#             max_new_tokens=128,
#             top_k=0,
#             top_p=1.0,
#             do_sample=True,
#         ),
#     ),
# )


# default_config.train.batch_size = 8
# default_config.train.total_steps = 1500
# default_config.train.checkpoint_dir = "checkpoints/sac_hh_125M"
# default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
# default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
# default_config.method.num_rollouts = 128
# default_config.train.tracker = None


# config = TRLConfig.update(default_config, {})

# # dataset = load_dataset("Dahoas/rm-static")
# # prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in dataset["train"]]
# # eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]
# # reward_fn = create_reward_fn()

# acceleratesactrainer = AccelerateSACTrainer(config)
# model = acceleratesactrainer.get_arch(config)
# freeze_bottom_causal_layers(model.base_model, config.model.num_layers_unfrozen)
# # print(model.eval())

# # trlx.train(
# #     prompts=prompts,
# #     eval_prompts=eval_prompts,
# #     reward_fn=reward_fn,
# #     config=config,
# #     stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
# # )
# # print(model.state_dict().keys())

# tokens = torch.randint(0, 10, size=[8, 10])
# attention_mask = torch.randn(8, 10)
# outputs = model(tokens, attention_mask, return_dict=True)

# v = ["q1_head", "q2_head", "q1_target_head", "q2_target_head"]
# for i in v:
#     print(i)
#     j = eval("model." + i + ".named_parameters()")
#     for name, para in j:
#         print(name, para.requires_grad)
# # print(model.q1_head.named_parameters())
# # print(model.q2_head.named_parameters())
# # print(model.q1_target_head.named_parameters())
# # print(model.q2_target_head.named_parameters())


x = torch.randn(8, 10)
x.requires_grad = True
print(x.requires_grad)

b = x.detach()
b[0, 0] = 100
print(x, b)
