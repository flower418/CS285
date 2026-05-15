"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128), # 表示隐藏层的维度
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []

        in_dim = state_dim # 观察到的输入是 5 个 dim
        out_dim = action_dim * chunk_size # 一下输出一个 chunk 的所有动作

        # 架构：FC + BN + ReLU
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(torch.relu())
            in_dim = hidden_dim

        # 最后一层直接 FC
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers) # 新写法


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.sample_actions(state)
        return nn.functional.mse_loss(action_chunk, pred)

    # 根据输入的 state 生成一段 action chunk
    def sample_actions(
        self,
        state: torch.Tensor, # (5,)，表示当前输入的维度，默认为 5 维
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred = self.net(state) # (action_chunk, 2)
        return pred.reshape(-1, self.chunk_size, self.action_dim) # (B, chunk, action_dim)

class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        raise NotImplementedError


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
