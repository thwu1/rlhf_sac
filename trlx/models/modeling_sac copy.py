import gc
import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from torchtyping import TensorType
from transformers.modeling_outputs import ModelOutput
from transformers.models.bloom import modeling_bloom
from transformers.models.opt import modeling_opt

from trlx.data.method_configs import MethodConfig, register_method
from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
    hf_get_decoder,
    hf_get_decoder_blocks,
    hf_get_decoder_final_norm,
    hf_get_hidden_size,
    hf_get_lm_head,
    hf_get_num_hidden_layers,
    make_head,
    whiten,
)

# KL Controllers


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


# SAC Configs


@dataclass
@register_method
class SACConfig(MethodConfig):
    """
    Config for SAC method

    :param sac_epochs: Number of updates per batch
    :type sac_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for SAC policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additional kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
    """

    sac_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    alpha: float
    gamma: float
    lam: float
    num_new_actions: int
    actor_coef: float
    adv_scale: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None

    def get_advantages_and_returns(
        self,
        values: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def get_tar_q_gae(self, logprobs_new, tar_q_new, rewards, ref_logprobs, old_logprobs, mask):
        lam, gamma, alpha = self.lam, self.gamma, self.alpha
        last_tar_delta = 0
        tar_delta_reversed = []
        assert logprobs_new.shape == mask.shape
        rewards = (rewards + alpha * (ref_logprobs - old_logprobs)) * mask
        tar_q_gae = (tar_q_new - alpha * logprobs_new) * mask
        for t in reversed(range(mask.shape[1])):
            next_q = tar_q_gae[:, t + 1] if t < mask.shape[1] - 1 else 0.0
            delta = rewards[:, t] + gamma * next_q - tar_q_gae[:, t]
            last_tar_delta = delta + gamma * lam * last_tar_delta
            tar_delta_reversed.append(last_tar_delta)
        tar_delta = torch.stack(tar_delta_reversed[::-1], dim=1)
        tar_q_gae = (tar_delta + tar_q_new + alpha * (old_logprobs - logprobs_new)) * mask
        return tar_q_gae.detach()

    def get_adv_gae(self, values, rewards, ref_logprobs, old_logprobs, mask):
        lam, gamma, alpha, use_whiten = self.lam, self.gamma, self.alpha, True
        n = mask.sum()
        last_adv = 0
        adv_reversed = []
        rewards = (rewards + alpha * (ref_logprobs - old_logprobs)) * mask
        values = values * mask
        for t in reversed(range(mask.shape[1])):
            next_v = values[:, t + 1] if t < mask.shape[1] - 1 else 0.0
            delta = rewards[:, t] + gamma * next_v - values[:, t]
            last_adv = delta + gamma * lam * last_adv
            adv_reversed.append(last_adv)
        adv = torch.stack(adv_reversed[::-1], dim=1)
        adv = (adv + alpha * old_logprobs) * mask
        adv_mean = adv.sum() / n
        adv_std = torch.sqrt((adv**2).sum() / n - adv_mean**2)
        if use_whiten:
            adv = mask * (adv - adv_mean) / (adv_std + 1e-8)
        return adv.detach(), {
            "adv_mean": adv_mean.item(),
            "adv_std": adv_std.item(),
            "adv_max": adv.max().item(),
            "adv_min": adv.min().item(),
        }

    def get_q(self, logprobs, values, mask):
        return (self.alpha * logprobs + values) * mask

    def get_next(self, x, mask):
        assert x.shape == mask.shape
        return torch.cat(
            [
                x[:, 1:] * mask[:, 1:],
                torch.zeros_like(x[:, -1]).unsqueeze(-1),
            ],
            dim=-1,
        )

    def loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        logprobs_new: TensorType["batch_size", "response_size"],
        values: TensorType["batch_size", "response_size"],
        values_tar: TensorType["batch_size", "response_size"],
        ref_logprobs: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        returns: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
        adv_logprobs: TensorType["batch_size", "response_size"],
        adv_tar_logprobs_new: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        n = mask.sum()

        log_ratio = (logprobs - ref_logprobs.detach()) * mask
        with torch.no_grad():
            approx_kl = (log_ratio.exp() - 1 - log_ratio).sum() / n
        values = values * mask
        rewards = rewards * mask

        q = self.get_q(adv_logprobs, values, mask)
        q_tar_new = self.get_q(adv_tar_logprobs_new, values_tar, mask)
        q_gae_tar = self.get_tar_q_gae(logprobs_new, q_tar_new, rewards, ref_logprobs, old_logprobs, mask)
        td = q - q_gae_tar
        q_loss = td.pow(2).sum() / n

        adv, adv_info = self.get_adv_gae(values_tar, rewards, ref_logprobs, old_logprobs, mask)
        actor_loss = ((log_ratio - self.adv_scale * adv) ** 2).sum() / n

        loss = q_loss + self.actor_coef * actor_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                q_loss=q_loss.item(),
                actor_loss=actor_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
            ),
            returns=get_tensor_stats(returns, mask, n),
            approx_kl=approx_kl.item(),
            padding_percentage=1 - n / mask.numel(),
            td=dict(
                max_td=td.max().item(),
                min_td=td.min().item(),
                mean_td=(td.sum() / n).item(),
                std_td=torch.sqrt((td**2).sum() / n - (td.sum() / n) ** 2).item(),
            ),
            adv=adv_info,
        )

        return loss, flatten_dict(stats)


# CausalLM architectures


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None
    value_tar: Optional[torch.FloatTensor] = None
    adv_logits: Optional[torch.FloatTensor] = None
    adv_tar_logits: Optional[torch.FloatTensor] = None


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["v_head", "v_tar_head", "adv_head", "adv_tar_head"]
    _supported_args = []

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
    ):
        super().__init__(base_model)
        self.v_head = make_head(hf_get_hidden_size(self.base_model.config), 1)
        self.v_tar_head = deepcopy(self.v_head).requires_grad_(False)
        self.adv_head = deepcopy(
            GPTModelBranch(
                self.base_model,
                num_layers_unfrozen=2,
            )
        ).requires_grad_(True)
        self.adv_tar_head = deepcopy(self.adv_head).requires_grad_(False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        outputs = self.base_model(**forward_kwargs)
        value = self.v_head(outputs.hidden_states[-1]).squeeze(-1)
        value_tar = self.v_tar_head(outputs.hidden_states[-1]).squeeze(-1)

        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value, value_tar, None, None)
            return outputs

        return CausalLMOutputWithValue(
            **outputs, value=value, value_tar=value_tar, adv_logits=None, adv_tar_logits=None
        )

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        base_model_state_dict = self.base_model.state_dict(*args, **kwargs)
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            base_model_state_dict[f"v_head.{k}"] = v
        v_tar_head_state_dict = self.v_tar_head.state_dict(*args, **kwargs)
        for k, v in v_tar_head_state_dict.items():
            base_model_state_dict[f"v_tar_head.{k}"] = v
        adv_head_state_dict = self.adv_head.state_dict(*args, **kwargs)
        for k, v in adv_head_state_dict.items():
            base_model_state_dict[f"adv_head.{k}"] = v
        adv_tar_head_state_dict = self.adv_tar_head.state_dict(*args, **kwargs)
        for k, v in adv_tar_head_state_dict.items():
            base_model_state_dict[f"adv_tar_head.{k}"] = v
        return base_model_state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        for k in list(state_dict.keys()):
            if "v_tar_head." in k:
                state_dict[k.replace("v_tar_head.", "")] = state_dict.pop(k)
        self.v_tar_head.load_state_dict(state_dict, strict=False)
        for k in list(state_dict.keys()):
            if "adv_head." in k:
                state_dict[k.replace("adv_head.", "")] = state_dict.pop(k)
        self.adv_head.load_state_dict(state_dict, strict=False)
        for k in list(state_dict.keys()):
            if "adv_tar_head." in k:
                state_dict[k.replace("adv_tar_head.", "")] = state_dict.pop(k)
        self.adv_tar_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()  # noqa: E702


class AutoModelForCausalLMWithHydraValueHead(AutoModelForCausalLMWithValueHead):
    _supported_modules = ["v_head", "frozen_head", "v_tar_head", "adv_tar_head", "adv_head"]
    _supported_args = ["num_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int = -1,
    ):
        super().__init__(base_model)
        self.num_layers_unfrozen = num_layers_unfrozen
        if self.num_layers_unfrozen > 0:
            config = self.base_model.config
            branch_class = hf_get_branch_class(config)
            # print("branch_class: ", branch_class)
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()
        # self.adv_tar_head = deepcopy(self.frozen_head).requires_grad_(False)

    def forward_adv(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True

        outputs = self.forward(**forward_kwargs)
        # Select the hidden state before the first branching layer
        input_hidden_state = outputs.hidden_states[-(self.num_layers_unfrozen + 1)]

        output_shape = outputs.hidden_states[-1].size()
        forward_kwargs.pop("input_ids", None)  # Ignore `input_ids` for branch head
        forward_kwargs.pop("inputs_embeds", None)  # Ignore `inputs_embeds` for branch head
        hydra_outputs = self.frozen_head(input_hidden_state, output_shape, **forward_kwargs)
        adv_logits = self.adv_head(input_hidden_state, output_shape, **forward_kwargs).logits
        adv_tar_logits = self.adv_tar_head(input_hidden_state, output_shape, **forward_kwargs).logits

        if not return_dict:
            return hydra_outputs.logits
        return CausalLMOutputWithValue(
            **hydra_outputs, value=None, value_tar=None, adv_logits=adv_logits, adv_tar_logits=adv_tar_logits
        )

    def forward_hydra(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True

        outputs = self.forward(**forward_kwargs)
        # Select the hidden state before the first branching layer
        input_hidden_state = outputs.hidden_states[-(self.num_layers_unfrozen + 1)]

        output_shape = outputs.hidden_states[-1].size()
        forward_kwargs.pop("input_ids", None)  # Ignore `input_ids` for branch head
        forward_kwargs.pop("inputs_embeds", None)  # Ignore `inputs_embeds` for branch head
        hydra_outputs = self.frozen_head(input_hidden_state, output_shape, **forward_kwargs)

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs


class ModelBranch(transformers.PreTrainedModel):
    """Implements the frozen upper trunk of the pretrained reference model used
    when computing the SAC KL-divergence penalty.
    """

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int,
    ):
        """
        Args:
            base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
            num_layers_unfrozen (int): The number of trainable layers
        """
        super().__init__(base_model.config)

        # The branch is defined by the last `num_layers_unfrozen` layers of the pretrained model
        decoder_blocks = deepcopy(hf_get_decoder_blocks(base_model))
        self.decoder_blocks = nn.ModuleList(list(decoder_blocks)[-num_layers_unfrozen:])
        self.final_norm = deepcopy(hf_get_decoder_final_norm(base_model))
        self.lm_head = deepcopy(hf_get_lm_head(base_model))

        self.hidden_size = hf_get_hidden_size(self.config)
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        # Freeze the entire branch
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class GPTModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/gpt2/modeling_gpt2.py#L743  # noqa: E501
        """
        batch_size, seq_length = hidden_states.shape[:2]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = hidden_states.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.decoder_blocks))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length)

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            kwargs = dict(
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # Assumes we are never training the branch
            block_params = inspect.getfullargspec(block.forward).args
            if "encoder_hidden_states" not in block_params:
                kwargs.pop("encoder_hidden_states")
                kwargs.pop("encoder_attention_mask")
            # Remove position_ids for GPT2Block
            if "position_ids" not in block_params:
                kwargs.pop("position_ids")

            outputs = block(hidden_states, **kwargs)

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class OPTModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/bdb84e2bada3658f99c6a81c963ec562f8485151/src/transformers/models/opt/modeling_opt.py#L840  # noqa: E501
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        input_shape = hidden_states.size()[:-1]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # `modeling_opt._make_causal_mask` @ transformers==4.27.1 doesn't have the `device` argument
            if "device" in inspect.getfullargspec(modeling_opt._make_causal_mask).args:
                kwargs = dict(device=hidden_states.device)
            else:
                kwargs = {}

            combined_attention_mask = modeling_opt._make_causal_mask(
                input_shape,
                hidden_states.dtype,
                past_key_values_length=past_key_values_length,
                **kwargs,
            ).to(hidden_states.device)

        if attention_mask is not None:
            expanded_attn_mask = modeling_opt._expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        attention_mask = combined_attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.decoder_blocks)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.decoder_blocks)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        # TODO: Add output projection support
        # https://github.com/huggingface/transformers/blob/699e90437f984d69ad3c9b891dd2e9d0fc2cffe4/src/transformers/models/opt/modeling_opt.py#L499  # noqa: E501
        # if self.project_out is not None:
        #     hidden_states = self.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        lm_logits = self.lm_head(hidden_states).contiguous()

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BloomModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/bloom/modeling_bloom.py#L623  # noqa: E501
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = hidden_states.shape[:2]

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.decoder_blocks))

        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = modeling_bloom.build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)

        combined_attention_mask = None
        device = attention_mask.device
        input_shape = (batch_size, seq_length)
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = modeling_bloom._make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        expanded_attn_mask = modeling_bloom._expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )
        causal_mask = combined_attention_mask

        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LlamaModelBranch(ModelBranch):
    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
                dim=-1,
            )
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, hidden_states, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                hidden_states.dtype,
                past_key_values_length=past_key_values_length,
            ).to(hidden_states.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]).to(
                hidden_states.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L491
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = hidden_states.shape[:2]
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else encoder_hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        lm_logits = self.lm_head(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Seq2Seq architectures


@dataclass
class Seq2SeqLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class AutoModelForSeq2SeqLMWithValueHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` sequence-to-sequence
    models that have a language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForSeq2SeqLM
    _supported_modules = ["v_head"]
    _supported_args = []

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
    ):
        super().__init__(base_model)
        self.v_head = make_head(hf_get_hidden_size(self.base_model.config), 1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutputWithValue:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        outputs = self.base_model(**forward_kwargs)
        last_hidden_state = outputs.decoder_hidden_states[-1]
        value = self.v_head(last_hidden_state).squeeze(-1)

        return Seq2SeqLMOutputWithValue(**outputs, value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        base_model_state_dict = self.base_model.state_dict(*args, **kwargs)
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            base_model_state_dict[f"v_head.{k}"] = v
        return base_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()  # noqa: E702


class AutoModelForSeq2SeqLMWithHydraValueHead(AutoModelForSeq2SeqLMWithValueHead):
    _supported_modules = ["v_head", "frozen_head"]
    _supported_args = ["num_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int = -1,
    ):
        super().__init__(base_model)
        self.num_layers_unfrozen = num_layers_unfrozen
        if self.num_layers_unfrozen > 0:
            branch_class = T5Branch  # TODO: Add support for other model branches
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()

    def forward_hydra(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutputWithValue:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        outputs = self.forward(**forward_kwargs)
        # Select the hidden state before the first branching layer
        input_hidden_state = outputs.decoder_hidden_states[-(self.num_layers_unfrozen + 1)]
        hydra_outputs = self.frozen_head(
            hidden_states=input_hidden_state,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=outputs.encoder_last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs


class T5Branch(ModelBranch):
    """Decoder only T5 branch"""

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int,
    ):
        super().__init__(base_model, num_layers_unfrozen=num_layers_unfrozen)
        self.dropout = hf_get_decoder(base_model).dropout
        self.is_decoder = True

    def forward(  # noqa: max-complexity
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/bc21aaca789f1a366c05e8b5e111632944886393/src/transformers/models/t5/modeling_t5.py#L899  # noqa: E501
        """
        batch_size, seq_length = hidden_states.shape[:2]
        input_shape = (batch_size, seq_length)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=hidden_states.device,
                dtype=torch.long,
            )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        for _, layer_module in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        sequence_output = hidden_states

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586  # noqa: E501
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,)

        return Seq2SeqLMOutputWithValue(
            logits=lm_logits,
            decoder_hidden_states=all_hidden_states,
            decoder_attentions=all_attentions,
        )


# Branch class utils


def hf_get_branch_class(
    config: transformers.PretrainedConfig,
) -> "ModelBranch":
    """Returns the model branch class for the given config."""
    gpt_branch_supported_archs = [
        "GPTJForCausalLM",
        "GPT2LMHeadModel",
        "GPTNeoForCausalLM",
        "GPTNeoXForCausalLM",
    ]
    opt_branch_supported_archs = ["OPTForCausalLM"]
    bloom_branch_supported_archs = ["BloomModel", "BloomForCausalLM"]
    llama_branch_supported_archs = ["LlamaModel", "LlamaForCausalLM"]
    arch = config.architectures[0]
    if arch in gpt_branch_supported_archs:
        return GPTModelBranch
    elif arch in opt_branch_supported_archs:
        return OPTModelBranch
    elif arch in bloom_branch_supported_archs:
        return BloomModelBranch
    elif arch in llama_branch_supported_archs:
        return LlamaModelBranch
    else:
        all_supported_archs = sum(
            [
                gpt_branch_supported_archs,
                opt_branch_supported_archs,
                bloom_branch_supported_archs,
                llama_branch_supported_archs,
            ],
            [],
        )
        raise ValueError(
            f"Unsupported architecture: `{arch}`. The following architectures are "
            f"available for model branching:\n{all_supported_archs}"
        )
