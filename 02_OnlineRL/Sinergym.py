#!/usr/bin/env python3.6

import gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation
from sinergym.utils.constants import RANGES_5ZONE

import argparse
import numpy as np
import torch
import os

import utils
import TD3

from numpy import array
import json

def eval_policy(policy, env_name, seed, eval_episodes=1):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	eval_env = NormalizeObservation(eval_env,ranges=RANGES_5ZONE)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Episode reward: {avg_reward}")
	print("---------------------------------------")
	return avg_reward



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Eplus-5Zone-mixed-continuous-stochastic-v1")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e5, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--alpha", default=2.5, type=float)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")


	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	for key,value in RANGES_5ZONE.items():
		print(f'{key} {value}')

	env = NormalizeObservation(env,ranges=RANGES_5ZONE)
	
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	print(f'State dim.:{state_dim} / Action dim.:{action_dim} / Max action:{max_action}')

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"alpha": args.alpha,
		"env": env,
		"batch_size": args.batch_size,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	print(f'{args} {kwargs}')

	def train_RL(env, policy, args, state_dim, action_dim, max_action, mode ='train_behavioral'):
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

		# Evaluate untrained policy
		evaluations = [eval_policy(policy, args.env, args.seed)]

		state, done = env.reset(), False

		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		for t in range(int(args.max_timesteps)):
			
			episode_timesteps += 1

			# Select action randomly or according to policy
			if t < args.start_timesteps and mode == 'train_behavioral':
				action = env.action_space.sample()
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

			# Perform action
			next_state, reward, done, _ = env.step(action) 
			done_bool = float(done) 

			# Store data in replay buffer
			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward

			# Train agent after collecting sufficient data
			if t >= args.start_timesteps and mode == 'train_behavioral':
				policy.train(replay_buffer, args.batch_size)

			if done: 
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset environment
				state, done = env.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1 

				if t > 0:
					evaluations.append(eval_policy(policy, args.env, args.seed))
					np.save(f"./results/{file_name}", evaluations)
					if args.save_model: policy.save(f"./models/{file_name}")

		if mode == 'generate_buffer':
			replay_buffer.save(f"./buffers/{args.env}")
		return policy

	behavioral_model = train_RL(
		env = env, 
		policy = policy, 
		args = args, 
		state_dim = state_dim, 
		action_dim = action_dim, 
		max_action = max_action, 
		mode ='train_behavioral')

	env.close()

	print(f'\n\nJob is finished.\n\n')