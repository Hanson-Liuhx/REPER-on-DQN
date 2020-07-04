import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym

from wrappers import *
from memory import ReplayMemory, NaivePrioritizedMemory, REPERMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import warnings

import argparse

warnings.filterwarnings("ignore", category=UserWarning)

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))


Transition_with_time = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward', 'time'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model_random(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




def optimize_model_PER(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    
    
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition_with_time(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    # print(state_batch.size())
    # exit()

    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # expected_state_action_values = (1 - done) * (next_state_values * GAMMA) + reward_batch
    
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Compute los by our own
    weights = torch.tensor(weights).float().to('cuda')

    loss = (state_action_values - expected_state_action_values)**2 * weights

    prios = loss + 1e-5


    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()

    memory.update_priorities(indices, prios)
    optimizer.step()




def optimize_model_REPER(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    
    
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    # print(state_batch.size())
    # exit()

    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # expected_state_action_values = (1 - done) * (next_state_values * GAMMA) + reward_batch
    
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Compute los by our own
    weights = torch.tensor(weights).float().to('cuda')

    loss = (state_action_values - expected_state_action_values)**2 * weights

    prios = loss + 1e-5


    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()

    memory.update_priorities(indices, prios)
    optimizer.step()



def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes, memory, render, sample):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            # REPER
            if sample == 'PER':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'), done)
            elif sample == 'random':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            elif sample == 'REPER':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'), done)


            state = next_state

            if steps_done > INITIAL_MEMORY:
                
                if sample == 'PER':
                    optimize_model_REPER(memory)
                elif sample == 'random':
                    optimize_model_random(memory)
                elif sample == 'PER':
                    optimize_model_PER(memory)

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return

def test(env, n_episodes, policy, render, sample_method):
    
    _time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 

    env = gym.wrappers.Monitor(env, './videos-{0}-{1}/dqn_pong_video'.format(_time, sample_method))
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default='PER', help="PER/REPER/random")

    args = parser.parse_args()
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    # initialize replay memory
    if args.sample_method == 'PER':
        # 2nd param default
        memory = NaivePrioritizedMemory(MEMORY_SIZE)

    elif args.sample_method == 'random':
        memory = ReplayMemory(MEMORY_SIZE)
    
    elif args.sample_method == 'REPER':
        memory = REPERMemory(MEMORY_SIZE)


    # train model
    train(env, 800, memory, render=False, sample=args.sample_method)
    torch.save(policy_net, "dqn_pong_model" + args.sample_method)
    policy_net = torch.load("dqn_pong_model" + args.sample_method)
    test(env, 1, policy_net, render=False, sample_method=args.sample_method)

