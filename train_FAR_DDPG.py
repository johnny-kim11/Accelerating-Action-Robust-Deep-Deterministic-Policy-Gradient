from utils import evaluate_policy
from datetime import datetime
from FAR_DDPG import FARDDPG_agent
import gym
import os, shutil
import argparse
import torch
import numpy as np


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')
parser.add_argument('--update_freq', type=int, default=50)

parser.add_argument('--alpha', type=float, default=0.1, help='alpha for the actor')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=2e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=5e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=1e4, help='random steps before trianing') # 1e4
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device


def main():
    EnvName = 'Walker2d-v4'
    threshold = 2400
    alpha = 0.0
    target_alpha = opt.alpha

    # Build Env
    env = gym.make(EnvName)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    eval_env = gym.make(EnvName)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    print(f'Env:{EnvName}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
        f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    folder_name = 'model' + '_' + str(opt.alpha)
    
    # Build DRL model
    if not os.path.exists(folder_name): os.mkdir(folder_name)
    agent = FARDDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    
    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s, info = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
        env_seed += 1
        done = False

        '''Interact & trian'''
        while not done:  
            if total_steps < opt.random_steps: a = env.action_space.sample()
            else: a = agent.select_action(s, deterministic=True)
            adv_a = agent.select_adv_action(s, alpha, agent.actor, agent.q_critic)
            input_action = (1-alpha)*a + alpha * adv_a + np.random.normal(0, opt.noise, size=action_dim)
            input_action  = input_action.clip(-max_action, max_action)
            s_next, r, dw, tr, info = env.step(input_action) # dw: dead&win; tr: truncated
            done = (dw or tr)

            agent.replay_buffer.add(s, input_action, r/100, s_next, dw)
            s = s_next
            total_steps += 1
            
            '''train'''
            if total_steps >= opt.random_steps and total_steps % opt.update_freq == 0:
                for _ in range(opt.update_freq):
                    agent.train(alpha)

            '''record & log'''
            if total_steps % opt.eval_interval == 0:
                ep_r = evaluate_policy(eval_env, agent, turns=5)
                if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{EnvName}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')
                if ep_r > threshold:
                    agent.save(int(total_steps/1000), folder_name)
                    
            if total_steps % 5000 == 0:
                alpha += 0.001
                if alpha > target_alpha: alpha = target_alpha
                print(f"curr alpha: {alpha}")

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()