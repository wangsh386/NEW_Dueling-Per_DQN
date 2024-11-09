######################################################################
# --------------------------------------------------------------
#对训练参数进行设置，并对基于DQN的无人机航迹规划算法模型进行训练
#----------------------------------------------------------------
# Set the training parameters and train the UAV track planning algorithm model based on DQN
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
from collections import deque
from replay_buffer import PrioritizedReplayBuffer, Transition
from  torch.autograd import Variable
import torch
from utils import LinearSchedule
import sys
import logging
import torch.optim as optim
import random
from model import QNetwork
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 检查当前系统是否支持CUDA
use_cuda = torch.cuda.is_available()

# 该变量用于存储浮点数张量
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# 使用GPU进行训练
device = torch.device("cuda" if use_cuda else "cpu")
# 导入模板类
from  torch.autograd import Variable

# 导入模板类
from replay_buffer import PrioritizedReplayBuffer, Transition

# 用于设置图形库的显示模式
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# 以下代码部分和上方代码的功能重合，需注释掉
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = torch.device("cuda" if use_cuda else "cpu")

# 批量大小
BATCH_SIZE = 128
TAU = 0.005
# 折扣率
gamma = 0.99
# 学习率(原0.0004)
LEARNING_RATE = 0.0005
# Q网络更新周期
TARGET_UPDATE = 10

# TODO 训练周期长度
num_episodes = 50000

print_every = 1
## 64 ## 16
hidden_dim = 16
# 贪心概率
min_eps = 0.01
# 最大贪心次数
max_eps_episode = 10

# beta 更新周期
beta_gain_steps = 2000
"""
# 状态空间维数
space_dim = 42
# 动作空间维度
action_dim = 27
"""

# 状态空间维数
space_dim = 91
# 动作空间维度
action_dim = 5
# 提示输出语句
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

# 阈值
threshold = 200
# 初始化各种参数（状态空间维数、动作空间维度、学习率）
env = Env(space_dim, action_dim, LEARNING_RATE)
# 提示输出语句
print('threshold: ', threshold)

# agent = Agent(space_dim, action_dim, hidden_dim)  #构造智能体

def epsilon_annealing(i_epsiode, max_episode, min_eps: float):

    # 计算斜率，用于在训练的过程中线性地降低目标的值
    slope = (min_eps - 1.0) / max_episode
    # 根据斜率和当前轮次计算目标值，并确保该值不会比min_eps更小
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps        

def save(directory, filename):

    # 将强化学习模型Q网络的参数保存到指定的文件中
    torch.save(env.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(env.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))

def run_episode(env, eps):

    """进行训练

    参数:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action        
        eps (float): eps-greedy for exploration

    返回:
        int: reward earned in this episode  返回回报值
    """

    # 环境重置
    state = env.reset()
    # 初始化奖励值
    total_reward = 0
    total_sensor = 0
    total_time = 0

    n_done = 0
    count = 0
    # 统计任务完成数量
    success_count = 0
    # 坠毁无人机数量
    crash_count = 0
    # 电量耗尽数量
    bt_count = 0
    # 超过最大步长的无人机
    over_count = 0

    while(1):
        count=count+1
        for i in range(len(env.uavs)):
            if env.uavs[i].done:
                # 无人机已结束任务，跳过
                continue
            # 根据Q值选取动作
            action = env.get_action(FloatTensor(np.array([state[i]])) , eps)

            # 根据选取的动作改变状态，获取收益
            next_state, reward, uav_done, info = env.step(action.detach(), i)

            # 求总收益
            total_reward += reward


            # TODO  PER DQN 需要在此整出个buffer


            # TODO PER DQN需要在此将优先级列表压入replay_memory
            # 存储交互经验
            env.replay_memory.add(state[i], action, reward, next_state, uav_done)

            """ if reward>0:
                # 正奖励，加强经验
                for t in range(2):
                    env.replay_memory.push(
                        (FloatTensor(np.array([state[i]])), 
                        action, # action is already a tensor
                        FloatTensor([reward]), 
                        FloatTensor([next_state]), 
                        FloatTensor([uav_done]))) """

            if info==1:
                success_count=success_count+1
            elif info==2:
                crash_count+=1
            elif info==3: 
                bt_count+=1
            elif info==5: 
                over_count+=1

            # 结束状态
            if uav_done:
                env.uavs[i].done=True
                n_done=n_done+1
                continue
            # 状态变更
            state[i] = next_state

        # env.render() 可视化显示当前环境状态
        if count % 5 == 0 and env.replay_memory.size > BATCH_SIZE:

            #batch = env.replay_memory.sample(BATCH_SIZE)

            # 训练Q网络
            env.learn(gamma,BATCH_SIZE)

        if n_done>=env.n_uav:
            break

        # plt.pause(0.001) 添加暂停操作

    for i in range(len(env.uavs)):
        total_sensor = total_sensor + env.uavs[i].collect_done

    for i in range(len(env.uavs)):
        total_time = total_time + env.uavs[i].collect_time + env.uavs[i].step

    if success_count >= 0.8*env.n_uav and env.level < 10:
        # 通过率较大，难度升级
        env.level = env.level+1
    return total_reward, total_sensor, total_time, [success_count, crash_count, bt_count, over_count]

def train():

    # 建立双端队列
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []
    time_start = time.time()

    # TODO 载入预训练模型（此处先不进行修改）

    check_point_Qlocal = torch.load('Qlocal.pth')
    check_point_Qtarget = torch.load('Qtarget.pth')
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    epoch = check_point_Qlocal['epoch']

    # epoch = 208500
    beta_scheduler = LinearSchedule(beta_gain_steps, 0.4, 1.0)  # 控制优先级采样中的重要度采样权重的增加
    run_step = 0
    for i_episode in range(num_episodes):
        # 计算贪心概率
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        # 运行一幕，获得得分,返回到达目标的个数
        score, total_sensor, total_time, info = run_episode(env, eps)
        # 跑完一个episode，更新run_step
        run_step += 1
        # 更新replay_memory的beta值
        env.replay_memory.beta = beta_scheduler.value(run_step)

        # 添加得分
        scores_deque.append(score)
        scores_array.append(score)
        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)
            
        if i_episode % print_every == 0 and i_episode > 0:
            logging.basicConfig(filename='output.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO)
            old_stdout = sys.stdout

            # 将标准输出重定向到日志文件
            sys.stdout = open('output.log', 'a')
            print(
                'sum_Episode: {:5} Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, eps-greedy: {:5.2f} Time: {:02}:{:02}:{:02} level:{:5} sersor-cellect-done:{:7} timecost:{:7} num_success:{:2}  num_crash:{:2}  num_none_energy:{:2}  num_overstep:{:2}'. \
                format(i_episode + epoch, i_episode, score, avg_score, eps, dt // 3600, dt % 3600 // 60, dt % 60,
                       env.level, total_sensor, total_time, info[0], info[1], info[2], info[3]))
            # 刷新并保存日志文件
            sys.stdout.flush()
            # 恢复原始的标准输出
            sys.stdout = old_stdout

        # 保存模型参数
        if i_episode % 100==0:
            # 每100周期保存一次网络参数
            state = {'model': env.q_target.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qtarget.pth")
            state = {'model': env.q_local.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qlocal.pth")

        if i_episode % TARGET_UPDATE == 0:
            env.q_target.load_state_dict(env.q_local.state_dict()) 
    
    return scores_array, avg_scores_array

  


if __name__ == '__main__':
    scores,avg_scores=train()
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))
