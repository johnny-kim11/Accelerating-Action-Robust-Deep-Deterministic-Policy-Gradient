import gym
import numpy as np
import ray
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from FAR_DDPG import FARDDPG_agent
import torch
import argparse

ray.init()

parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
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
parser.add_argument('--random_steps', type=int, default=1e4, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')

EnvName = 'Walker2d-v4'
threshold = 2000
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt.dvc)

# Build Env
env = gym.make(EnvName)
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

import os

env_name = "Walker2d-v4"
alpha = 0.2

path = './model_' + str(alpha) + '/actor/'
file_names  = os.listdir(path)

target_lst = []
for i in range(len(file_names)):
    target_lst.append(file_names[i][:-4])
    
print(target_lst)
    
@ray.remote
def evaluate_arrl(env, mass_param, fric_param, agent):
    times = 10  # Perform three evaluations and calculate the average
    env.model.body_mass = env.model.body_mass * mass_param
    env.model.geom_friction = env.model.geom_friction * fric_param
    evaluate_reward = 0
    for t in range(times):
        s = env.reset()[0]
        done = False
        episode_reward = 0
        timestep = 0

        for t in range(1000):
            a_real = agent.select_action(s, True)
            s_, r, t1, t2, _ = env.step(a_real)

            done = t1 or t2
            if done:
                break
            episode_reward += r
            s = s_
            timestep += 1
        evaluate_reward += episode_reward
    return int(evaluate_reward / times)




mass_start = 10
mass_end = 231
mass_step =  int((mass_end-1-mass_start)/10)

friction_start = 10
friction_end = 901
friction_step = int((friction_end-1-friction_start)/10)

threshold = 2000

for k in range(len(target_lst)):
    print(f"{k+1}/{len(target_lst)}")
    arrl_agent = FARDDPG_agent(**vars(opt))
    
    folder_name = 'model' + '_' + str(opt.alpha)
    arrl_agent.load(target_lst[k], folder_name)
    
    env = gym.make(env_name)
    score_lst = []

    for friction in range(friction_start, friction_end, friction_step): # 250, 25
        score = [evaluate_arrl.remote(env, mass_param/100, friction/100, arrl_agent) for mass_param in range(mass_start, mass_end, mass_step)]
        print(ray.get(score))
        score_lst.append(ray.get(score))

    score_np = np.array(score_lst)
    count = np.sum(score_np >= threshold)
    print(count)
    if count < 10:
        print(path + target_lst[k] + '.pth')
        os.remove(path + target_lst[k] + '.pth')
        print(f"Removed ")
        continue

    mass_mem = [str(round(param/100, 2)) for param in range(mass_start, mass_end, mass_step)]
    friction_mem = [str(round(param/100, 2)) for param in range(friction_start, friction_end, friction_step)] # range(5, 16)] #   
    df_proposed1 = pd.DataFrame(score_np).T
    df_proposed1.columns = friction_mem    
    df_proposed1.index = mass_mem

    ax1 = sns.heatmap(df_proposed1, vmax=threshold, vmin=0)
    ax1.set_xticklabels(friction_mem, rotation=60)
    ax1.set_yticklabels(mass_mem, rotation=0)
    ax1.set_xlabel("friction coef", fontsize=15)
    ax1.set_ylabel("Mass coef", fontsize=15)
    ax1.invert_yaxis()

    fig_name = "./result_pics/" + str(alpha) + '/' + file_names[k][:-4]+ '_'+str(count) + ".png"
    plt.title(env_name + ' (Proposed)', fontsize=20)
    plt.savefig(fig_name) 
    print(fig_name)
    plt.clf()
    