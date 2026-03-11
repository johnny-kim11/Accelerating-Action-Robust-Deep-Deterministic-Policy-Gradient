from utils import Actor, Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy
from itertools import product

def generate_combinations(n):
    values = [-1, 1]
    combinations = list(product(values, repeat=n))
    return torch.tensor(combinations, dtype=torch.float)

class FARDDPG_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)

		self.adv_candidate = generate_combinations(self.action_dim).to(self.dvc)
		self.adv_candidate_batch = generate_combinations(self.action_dim).to(self.dvc).repeat(self.batch_size,1)
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)
	def select_adv_action(self, state, alpha, actor_old, critic_old):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = actor_old(state) # from [[x,x,...,x]] to [x,x,...,x]
			a = a.repeat(2**self.action_dim,1)
			input_action_sample = (1-alpha)*a + alpha*self.adv_candidate
			state = state.repeat(2**self.action_dim,1)
			q_values = critic_old(state, input_action_sample)
			a_idx = torch.argmin(q_values).item()
		return self.adv_candidate[a_idx].cpu().numpy()
	
	def adv_batch(self, state, alpha, actor, critic):
		with torch.no_grad():
			state = torch.unsqueeze(state, 1)
			state = state.repeat(1,2**self.action_dim,1)

			a = actor(state).view(-1,self.action_dim)
			adv_action = self.adv_candidate_batch
			adv_batch = (1-alpha)*a + alpha*adv_action
			state = state.view(-1,self.state_dim)
			q_values = critic(state, adv_batch)

			value = q_values.view(self.batch_size,2**self.action_dim)
			value = value.argmin(dim=1).unsqueeze(-1)
			adv_candidate = self.adv_candidate_batch.view(self.batch_size, 2**self.action_dim, self.action_dim)

			result = torch.gather(adv_candidate, dim=1, index=value.unsqueeze(-1).expand(adv_candidate.size(0), -1, adv_candidate.size(-1)))
			return result.squeeze(1)

	def train(self, alpha):
		batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(self.batch_size)
		for params in self.q_critic.parameters():
			params.requires_grad = False

		with torch.no_grad():
			state = torch.unsqueeze(batch_s, 1)
			state = state.repeat(1,2**self.action_dim,1)
			adv_action = self.adv_candidate_batch

		a = self.actor(state).view(-1,self.action_dim)
		adv_batch = (1-alpha)*a + alpha*adv_action
		state = state.view(-1,self.state_dim)
		q_values = self.q_critic(state, adv_batch)

		value = q_values.view(self.batch_size,2**self.action_dim)
		Q_ = value.min(dim=1)[0].unsqueeze(-1)
		actor_loss = -Q_.mean() 

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
	
		for params in self.q_critic.parameters():
			params.requires_grad = True

		with torch.no_grad():
			state = torch.unsqueeze(batch_s_, 1)
			state = state.repeat(1,2**self.action_dim,1)

			a = self.actor_target(state).view(-1,self.action_dim)
			adv_action = self.adv_candidate_batch
			adv_batch = (1-alpha)*a + alpha*adv_action
			state = state.view(-1,self.state_dim)
			q_values = self.q_critic_target(state, adv_batch)

			value = q_values.view(self.batch_size,2**self.action_dim)
			Q_ = value.min(dim=1)[0].unsqueeze(-1)

			target_Q = batch_r + (~batch_dw) * self.gamma * Q_
    
		current_Q = self.q_critic(batch_s, batch_a)
		critic_loss = F.mse_loss(current_Q, target_Q)

		self.q_critic_optimizer.zero_grad()
		critic_loss.backward()
		self.q_critic_optimizer.step()
    
		# Update the frozen target models
		with torch.no_grad():
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,timestep, path):
		torch.save(self.actor.state_dict(), "./"+path+"/actor/{}.pth".format(timestep))
		torch.save(self.q_critic.state_dict(), "./"+path+"/critic/{}.pth".format(timestep))

	def load(self,timestep, path):
		self.actor.load_state_dict(torch.load("./"+path+"/actor/{}.pth".format(timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./"+path+"/critic/{}.pth".format(timestep), map_location=self.dvc))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]