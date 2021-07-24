import numpy as np
from collections import namedtuple
import random


from collections import namedtuple


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
        self.Transition = namedtuple(
            'Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = self.Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
