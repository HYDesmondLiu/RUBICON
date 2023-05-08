import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinergym
from typing import List, Any, Sequence
import sinergym
from sinergym.utils.common import get_season_comfort_range
from sinergym.utils.constants import RANGES_5ZONE
from sinergym.utils.controllers import RBC5Zone
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class RuleBasedController(RBC5Zone):

	def get_datetime(self, row):
		year, month, day = row
		return datetime(year, month, day)

	def get_season_comfort_range_rbc(self, row):
		year, month, day = row
		return get_season_comfort_range(year, month, day)

	def act_batch_vectorization(self, observation, RANGES_5ZONE_STATE, action_min, action_max, batch_size):
		""" 
			year
			month 
			day 
			hour
		    Zone Thermostat Heating Setpoint Temperature(SPACE1-1)
			Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)
			Zone Air Temperature(SPACE1-1)
		"""
		observation_rbc = observation.index_select(1, torch.LongTensor([0,1,2,3,10,11,12]).to(device))
		
		year  = observation_rbc[:,0].detach().cpu().reshape(-1,1)
		month = observation_rbc[:,1].detach().cpu().reshape(-1,1)
		day   = observation_rbc[:,2].detach().cpu().reshape(-1,1)
		hour  = observation_rbc[:,3].detach().cpu().reshape(-1,1)

		# time consists of [year month day]
		time = observation_rbc[:,:3].detach().cpu().numpy()

		current_heat_setpoint = observation_rbc[:,4].detach().cpu().reshape(-1,1).numpy()
		current_cool_setpoint = observation_rbc[:,5].detach().cpu().reshape(-1,1).numpy()
		in_temp = observation_rbc[:,6].detach().cpu().reshape(-1,1).numpy()

		# Update setpoints
		new_heat_setpoint = copy.deepcopy(current_heat_setpoint)
		new_cool_setpoint = copy.deepcopy(current_cool_setpoint)

		season_comfort_range = np.apply_along_axis(self.get_season_comfort_range_rbc, 1, time)
		
		in_temp_less_szn_comfort_range_low = np.less(in_temp, season_comfort_range[:,0].reshape(-1,1))

		indices_less = [in_temp_less_szn_comfort_range_low.nonzero()[0]]
		if indices_less[0].size > 0:
			new_heat_setpoint[indices_less] = current_heat_setpoint[indices_less] + 1
			new_cool_setpoint[indices_less] = current_cool_setpoint[indices_less] + 1

		in_temp_greater_szn_comfort_range_high = np.greater(in_temp, season_comfort_range[:,1].reshape(-1,1))
		indices_greater = [in_temp_greater_szn_comfort_range_high.nonzero()[0]]		
		if indices_greater[0].size > 0:
			new_cool_setpoint[indices_greater] = current_cool_setpoint[indices_greater] - 1
			new_heat_setpoint[indices_greater] = current_heat_setpoint[indices_greater] - 1

		action = np.concatenate([new_heat_setpoint, new_cool_setpoint],1)

		current_dt = np.apply_along_axis(self.get_datetime, 1, time)
		weekday_list = np.array([x.weekday() for x in current_dt])
		weekend = np.where(weekday_list > 5, weekday_list, 0).nonzero()[0]
		off_hours = np.concatenate([np.where(hour>22, hour, 0).nonzero()[0],
			np.where(hour<6, hour, 0).nonzero()[0]])
		weekend_off = np.concatenate( [weekend, off_hours])
		if len(weekend_off) > 0:
			action[weekend] = [18.33, 23.33]
		action = torch.FloatTensor(action).to(device)
		action = (action - action_min) / (action_max - action_min).clamp(-1,1)
		return action


class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		env,
		seed,
		batch_size = 256,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		# RBC regularization
		self.env = env
		self.rbc = RuleBasedController(env)
		self.RANGES_5ZONE_STATE = None
		self.state_range_max = None
		self.state_range_min = None		
		self.action_dim = action_dim
		
		self.heating_stpt_range = self.env.ranges['Zone Thermostat Heating Setpoint Temperature(SPACE1-1)']
		self.cooling_stpt_range = self.env.ranges['Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)']
		heating_stpt_min = torch.FloatTensor([self.heating_stpt_range[0]]).repeat(batch_size, 1).to(device)
		heating_stpt_max = torch.FloatTensor([self.heating_stpt_range[1]]).repeat(batch_size, 1).to(device)
		cooling_stpt_min = torch.FloatTensor([self.cooling_stpt_range[0]]).repeat(batch_size, 1).to(device)
		cooling_stpt_max = torch.FloatTensor([self.cooling_stpt_range[1]]).repeat(batch_size, 1).to(device)

		self.action_min = torch.cat((heating_stpt_min, cooling_stpt_min),1)
		self.action_max = torch.cat((heating_stpt_max, cooling_stpt_max),1)
		print(f'Heating stpt range: {self.heating_stpt_range} Cooling stpt range: {self.cooling_stpt_range}')

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def inverse_state_normalization(self, state, batch_size):
		env_obs = self.env.variables['observation']
		env_obs_range = list(RANGES_5ZONE.keys())
		
		if self.total_it == 1:
			self.RANGES_5ZONE_STATE = { key: RANGES_5ZONE[key] for key in env_obs }
			self.state_range_min = torch.FloatTensor([x[0] for x in list(self.RANGES_5ZONE_STATE.values())]).repeat(batch_size,1).to(device)
			self.state_range_max = torch.FloatTensor([x[1] for x in list(self.RANGES_5ZONE_STATE.values())]).repeat(batch_size,1).to(device)

		raw_state = state * (self.state_range_max - self.state_range_min) + self.state_range_min

		return raw_state


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		critic_pre = dict(self.critic.named_parameters())

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# De-normalization for RBC
		state_raw = self.inverse_state_normalization(state, batch_size)
		
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			# RBC regularization
			pi_rbc = self.rbc.act_batch_vectorization(state_raw, self.RANGES_5ZONE_STATE, 
				self.action_min, self.action_max, batch_size, ).clamp(-1, 1)

			Q_rbc = self.critic.Q1(state, pi_rbc)
			Q_buffer = self.critic.Q1(state, action)

			if Q_buffer.mean() >= Q_rbc.mean():
				actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
			else:
				actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, pi_rbc)

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)