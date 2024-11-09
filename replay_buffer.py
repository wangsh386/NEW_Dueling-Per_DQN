
######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

import random
import numpy as np
from collections import namedtuple
from utils import SumTree
import torch
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class PrioritizedReplayBuffer(object):
    def __init__(self, maxsize):
        self.ptr = 0
        self.size = 0

        max_size = maxsize
        self.max_size = max_size
        self.state = np.zeros((max_size, 91))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, 91))
        self.dw = np.zeros((max_size, 1))


        self.sum_tree = SumTree(max_size)
        self.alpha = 0.6
        self.beta = 0.4

        self.device = device

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action.item()  # TODO action 是张量，要将其改为int类型值
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
        priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
        self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  # 更新当前经验在sum_tree中的优先级

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)

        return (
            torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
            ind,
            Normed_IS_weight.to(self.device)  # shape：(batch_size,)
        )

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update_priority(data_index=index, priority=priority)
