# Torch
import torch
from torch import nn
from torch.utils.data import DataLoader

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, ToPILImage, Resize, InterpolationMode

import torch.nn.functional as F

# Gym
import gym
from PIL import Image

# Misc
import os
import sys
import math
import random
import numpy as np
from itertools import count
from collections import namedtuple, deque

# Configs
config = {
    # General
    'DEVICE':        'cuda' if torch.cuda.is_available() else 'cpu',
    'EPOCHS':        500,
    'BATCH_SIZE':    128,
    'MEMORY_SIZE':   10000,
    'RUN_NAME':      'lastrun',

    # Trainer
    'GAMMA':         0.999,
    'EPS_START':     0.9,
    'EPS_END':       0.05,
    'EPS_DECAY':     200,
    'TARGET_UPDATE': 10,

    # Directories, not used by :NeuralNetwork
    'RUNS_DIR':      'runs',
    'DATA_DIR':      'data',
    'MODELS_DIR':    'models'
}

Transition = namedtuple(
    'Transition',
    (
        'state',
        'action',
        'next_state',
        'reward'
    )
)

class ReplyMemory(object):
    def __init__(self, size):
        self.memory = deque([], maxlen = size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Environment(object):
    def __init__(self, config, make = 'CartPole-v0'):
        self.config = config
        self.env = gym.make(make)
        self.observation = self.env.reset()
        self.action_space = self.env.action_space

        self.screen_transform = Compose([
            ToPILImage(),
            Resize(40, interpolation = InterpolationMode.BOX),
            ToTensor()
        ])

        self.get_screen()
        self.height, self.width = self.screen.shape[2:]

    def reset(self):
        self.env.reset()

    def step(self, item):
        return self.env.step(item)

    def get_screen(self):
        self.screen = self.env.render(mode = 'rgb_array').transpose((2, 0, 1))

        _, self.height, self.width = self.screen.shape
        self.screen = self.screen[:, int(self.height * 0.4):int(self.height * 0.8)]
        self.view_width = int(self.width * 0.6)

        cart_location = self.__cart_location()

        if cart_location < self.view_width // 2:
            slice_range = slice(self.view_width)
        elif cart_location > (self.width - self.view_width // 2):
            slice_range = slice(-self.view_width, None)
        else:
            slice_range = slice(cart_location - self.view_width // 2,
                                cart_location + self.view_width // 2)

        self.screen = self.screen[:, :, slice_range]

        self.screen = np.ascontiguousarray(self.screen, dtype = np.float32) / 255
        self.screen = torch.from_numpy(self.screen)
        self.screen = self.screen_transform(self.screen).unsqueeze(0)

        return self.screen

    def __cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.width / world_width

        return int(self.env.state[0] * scale + self.width / 2.0)

class DQN(nn.Module):
    def __init__(self, w, h, outputs, config, writer = None):
        super(DQN, self).__init__()

        self.config = config
        self.writer = writer
        self.device = self.config['DEVICE']
        self.width, self.height = w, h
        self.outputs = outputs

        self.Stack = nn.Sequential(
            nn.Conv2d( 3, 16, kernel_size = 5, stride = 2), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 5, stride = 2), nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(self.__output_size(self.width, self.height), outputs)
        )

        self.Stack.apply(self._init_weights)

        super(DQN, self).to(self.device)

    def forward(self, x):
        return self.Stack(x)

    def save(self, path):
        dirpath = os.path.dirname(path)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        torch.save(super(DQN, self).state_dict(), path)

    def load(self, path):
        super(DQN, self).load_state_dict(torch.load(path))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __output_size(self, w, h, kernel_size = 5, stride = 2):
        def size_out(size):
            return (size - (kernel_size - 1) - 1) // stride + 1

        for i in range(3):
            w = size_out(w)
            h = size_out(h)

        return w * h * 32

class Network(object):
    def __init__(self, config, writer = None):
        self.config = config
        self.writer = writer
        self.device = self.config['DEVICE']

        self.env = Environment(self.config, 'CartPole-v0')
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(self.env.width, self.env.height, self.n_actions, config, writer)
        self.target_net = DQN(self.env.width, self.env.height, self.n_actions, config, writer)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplyMemory(self.config['MEMORY_SIZE'])

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.__eps_threshold()

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device = self.device, dtype=torch.long)

    def optimize(self):
        if len(self.memory) < self.config['BATCH_SIZE']:
            return

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions))

        nfinal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device = self.device, dtype = torch.bool
        )

        nfinal_next = torch.cat([s for s in batch.next_state if s is not None])
        bstate  = torch.cat(batch.state)
        baction = torch.cat(batch.action)
        breward = torch.cat(batch.reward)

        avalues = self.policy_net(bstate).gather(1, baction)

        state_next_values = torch.zeros(self.config['BATCH_SIZE'], device = self.device)
        state_next_values[nfinal_mask] = self.target_net(nfinal_next).max(1)[0].detach()

        state_action_pred = (state_next_values * self.config['GAMMA']) + breward

        loss = self.loss_function(avalues, state_action_pred.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        for i in range(self.config['EPOCHS']):
            print(f"Epoch {i+1}")

            self.env.reset()

            screen_last = self.env.get_screen()
            screen_curr = self.env.get_screen()
            state = screen_curr - screen_last
    
            for t in count():
                action = self.select_action(state)
    
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device = self.device)
    
                screen_last = screen_curr
                screen_curr = self.env.get_screen()
                if not done:
                    state_next = screen_curr - screen_last
                else:
                    state_next = None
    
                self.memory.push(state, action, state_next, reward)
                state = state_next
    
                self.optimize()
                if done:
                    break

        if i % self.config['TARGET_UPDATE'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.env.env.render()
        self.env.env.close()

    def save(self, path):
        self.target_net.save(path + '_target')
        self.policy_net.save(path + '_policy')

    def load(self, path):
        self.target_net.load(path + '_target')
        self.policy_net.load(path + '_policy')

    def __eps_threshold(self):
        return self.config['EPS_END'] + (self.config['EPS_START'] - \
               self.config['EPS_END']) * math.exp(-1. * self.steps_done / self.config['EPS_DECAY'])

if __name__ == '__main__':
    logdir = os.path.join(config['RUNS_DIR'], config['RUN_NAME'])
    modfile = os.path.join(config['MODELS_DIR'], config['RUN_NAME'])

    writer = SummaryWriter(log_dir = logdir)
    model = Network(config, writer)

    if os.path.isfile(modfile):
        model.load(modfile)

    try:
        model.train()
        writer.flush()
        writer.close()
    except KeyboardInterrupt:
        model.save(modfile)
        writer.flush()
        writer.close()
