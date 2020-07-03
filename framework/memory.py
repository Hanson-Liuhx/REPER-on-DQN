import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        ret = random.sample(self.memory, batch_size)
        # A list of BATCH_SIZE trasition object, each of them: ('state', 'action', 'next_state', 'reward'))
        return ret
    
    def __len__(self):
        return len(self.memory)


# class PrioritizedReplay(object):
#     def __init__(self, capacity):
#         pass



# From http://10.15.89.41:30303/notebooks/code_from_jupyter/full-version/RL-Adventure/4.prioritized%20dqn.ipynb
# Not origin web... fix later
class NaivePrioritizedMemory(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.memory     = []
        self.position   = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state)
        
        self.priorities[self.position] = max_prio

        # TODO Maybe another way
        self.position = (self.position + 1) % self.capacity

    
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        # Stardardized formula
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        # print(self.priorities)
        # exit()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()        

        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)





class REPERMemory(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.memory     = []
        self.position   = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state)
        
        self.priorities[self.position] = max_prio

        # TODO Maybe another way
        self.position = (self.position + 1) % self.capacity

    
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        # Stardardized formula
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        # print(self.priorities)
        # exit()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()        

        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)