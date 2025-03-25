import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
import argparse
import os

import utils
#import TD3
#import dyna_gamid
import dyna_gamid_vae
import gamid_vae2
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(time_step, policy, env_name, seed, actor_array, eval_episodes=10):
	eval_env = gym.make(env_name, render_mode=None)
	eval_env.reset(seed=seed + 100)
	rewards = []
	for _ in range(eval_episodes):
		state, info = eval_env.reset()
		episode_reward = 0
		done = truncated = False
		while not (done or truncated):
			with torch.no_grad():
				action = policy.select_action(np.array(state))
			state, reward, done, truncated, info = eval_env.step(action)
			episode_reward += reward
		rewards.append(episode_reward)
	avg_reward = np.mean(rewards)
	print("---------------------------------------")
	if isinstance(policy, dyna_gamid_vae.DynamicGamidVae) or isinstance(policy, gamid_vae2.GamidVae2):
		cur_actor = len(policy.actors)
		actor_array.append(cur_actor)
		print(f"Evaluation over {eval_episodes} episodes and {time_step} steps: {avg_reward:.3f} Current actor: {cur_actor}")
		# np.save(f"{actors_num_dir}/{file_name}", actor_array)
	else:
		print(f"Evaluation over {eval_episodes} episodes and {time_step} steps: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v4")          # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default="False", type=bool)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--file_append", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}" if args.file_append == "" else f"{args.policy}_{args.env}_{args.seed}_{args.file_append}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	results_dir = f"./results/{args.policy}"
	actors_num_dir = f"./actor_num_results/{args.policy}"
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	if not os.path.exists(actors_num_dir):
		os.makedirs(actors_num_dir)

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	'''
	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	'''
	# For newer Gymnasium versions (>=0.26.0)
	env = gym.make(args.env, render_mode=None)  # The second return value is the step metadata
	rng = np.random.default_rng(args.seed)  # Create a random number generator
	env.reset(seed=args.seed)  # Set the seed during reset
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		#policy = TD3.TD3(**kwargs)
	elif args.policy == "DynaGamid":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		#policy = dyna_gamid.DynamicGamid(**kwargs)
	elif args.policy == 'DynaGamidVae':
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = dyna_gamid_vae.DynamicGamidVae(**kwargs)
	elif args.policy == 'GamidVae2':
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = gamid_vae2.GamidVae2(**kwargs)
	'''
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	'''

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	actor_array = [3]
	evaluations = [eval_policy(0, policy, args.env, args.seed, actor_array)]	

	state, info = env.reset()
	done = False
	truncated = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, truncated, info = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done or truncated: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			# print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, info = env.reset()
			done = False
			truncated = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(t, policy, args.env, args.seed, actor_array))
			np.save(f"{results_dir}/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")

			
