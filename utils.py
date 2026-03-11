import torch.nn.functional as F
import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, act='silu', softplus_beta=10.0):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

        if act == 'silu':
            self.act = nn.SiLU()            # 추천 1: 가볍고 C^2
        elif act == 'softplus':
            self.act = nn.Softplus(beta=softplus_beta)  # 추천 2: ReLU 근사+매끈
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'mish':
            self.act = nn.Mish()
        else:
            raise ValueError("act must be one of {'silu','softplus','gelu','mish'}")

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = self.act(self.l1(sa))
        q = self.act(self.l2(q))
        q = self.l3(q)    # 출력층은 보통 선형 유지
        return q

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)