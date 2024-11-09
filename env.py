######################################################################
# Environment build
#env类对城市环境进行三维构建与模拟，利用立方体描述城市建筑，
# 同时用三维坐标点描述传感器。对环境进行的空间规模、风况、无人机集合、
# 传感器集合、建筑集合、经验池进行初始化设置，并载入DQN神经网络模型。
# env类成员函数能实现UAV行为决策、UAV决策经验学习、环境可视化、单时间步推演等功能。
#----------------------------------------------------------------
# The env class constructs and simulates the urban environment in 3D, 
# uses cubes to describe urban buildings, and uses 3D coordinate points 
# to describe sensors. Initialize the environment's spatial scale, wind conditions,
# UAV collection, sensor collection, building collection, and experience pool, 
# and load the DQN neural network model. 
# The env class member function can implement UAV behavioral decision-making,
# UAV decision-making experience learning, environment visualization, and single-time-step deduction.
##############################################################################
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import random
from model import QNetwork, ConvDuelingDQN, DuelingDQN
from UAV import *
from  torch.autograd import Variable
from replay_buffer import PrioritizedReplayBuffer, Transition

# 检查当前系统是否支持CUDA
use_cuda = torch.cuda.is_available()

# 该变量用于存储浮点数张量
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# 使用GPU进行训练
device = torch.device("cuda" if use_cuda else "cpu")

class building():
    def __init__(self, x, y, l, w, h):
        self.x=x   # 建筑中心x坐标
        self.y=y   # 建筑中心y坐标
        self.l=l   # 建筑长半值
        self.w=w   # 建筑宽半值
        self.h=h   # 建筑高度

class sn():
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.data = data

class destination():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Env(object):

    def __init__(self,n_states,n_actions,LEARNING_RATE):

        # 定义规划空间大小
        self.len = 80
        self.width = 20
        # self.h=22
        # self.map=np.zeros((self.len,self.width,self.h))
        self.map = np.zeros((self.len, self.width))

        # 无人机对象集合
        self.uavs = []

        # 基站对象集合
        self.base = []

        # 设备对象集合
        self.equi = []

        # 建筑集合
        # self.bds=[]

        # 无人机对象
        self.target = []

        # 训练环境中的无人机个数
        self.n_uav = 15
        # 训练环境中的设备个数
        self.n_equi = 60
        # 无人机可控风速
        self.v0 = 40
        self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        self.ax = self.fig.add_subplot(1, 1, 1)
        # 打开绘图交互模式
        plt.ion()
        # 训练难度等级(0-10)
        self.level = 1

        # 神经网络参数

        # 初始化Q网络
        self.q_local = DuelingDQN(n_states, n_actions).to(device)
        # 初始化目标Q网络
        self.q_target = DuelingDQN(n_states, n_actions).to(device)

        # 损失函数：均方误差
        self.mse_loss = torch.nn.MSELoss()
        # 设置优化器，使用adam优化器
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)
        # 状态空间数目
        self.n_states = n_states
        # 动作集数目
        self.n_actions = n_actions
        # 初始化经验池
        self.replay_memory = PrioritizedReplayBuffer(100000)

    def get_action(self, state, eps, check_eps = True):

        global steps_done
        # 生成 0 - 1 的随机浮点数
        sample = random.random()

        if check_eps == False or sample > eps:
            with torch.no_grad():
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)   # 根据Q值选择行为
        else:
           ## return LongTensor([[random.randrange(2)]])
           return torch.tensor([[random.randrange(self.n_actions)]], device=device)   # 随机选取动作

    # TODO 对learn函数进行修改
    def learn(self, gamma, BATCH_SIZE):

        """准备训练

        参数:
        experiences (List[Transition]): batch of `Transition`   
        gamma (float): Discount rate of Q_target  折扣率
        """
        # 一次抽BATCH_SIZE个数据训练，不够就返回
        if self.replay_memory.size < BATCH_SIZE:
            return

        # 获取批量经验数据
        states, actions, rewards, next_states, dones, ind, Normed_IS_weight = self.replay_memory.sample(BATCH_SIZE)

        argmax_a = self.q_local(next_states).argmax(dim=1).unsqueeze(-1)  # 本地网络找到下一个状态动作
        max_q_prime = self.q_target(next_states).gather(1, argmax_a)  # 目标网络下一个状态最大Q值

        Q_targets = rewards + (1 - dones) * gamma * max_q_prime  # 更新Q目标值

        Q_expected = self.q_local(states).gather(1, actions)     # 获得Q估计值

        td_errors = (Q_expected - Q_targets).squeeze(-1)  # shape：(batch_size,)
        loss = (Normed_IS_weight * (td_errors ** 2)).mean()

        # 更新经验回放缓冲区中批量样本的优先级
        self.replay_memory.update_batch_priorities(ind, td_errors.detach().cpu().numpy())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.soft_update(self.q_local, self.q_target, 0.005)
               

    def soft_update(self, local_model, target_model, tau):

        # tau (float): interpolation parameter ,不妨令其为0.005

        # 更新Q网络与Q目标网络
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)     
            
    def hard_update(self, local, target):

        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(param.data)


    """
    def render(self,flag=0):
        # 绘制封闭立方体
        # 参数
        # x,y,z立方体中心坐标
        # dx,dy,dz 立方体长宽高半长
        # fig = plt.figure()

        if flag==1:
            # 第一次渲染，需要渲染建筑
            z=0
            # ax = self.fig.add_subplot(1, 1, 1, projection='3d')
            for ob in self.bds:
                # 绘画出所有建筑
                x=ob.x
                y=ob.y
                z=0
                dx=ob.l 
                dy=ob.w 
                dz=ob.h 
                xx = np.linspace(x-dx, x+dx, 2)
                yy = np.linspace(y-dy, y+dy, 2)
                zz = np.linspace(z, z+dz, 2)

                xx2, yy2 = np.meshgrid(xx, yy)

                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
            

                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x-dx), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)

                xx2, zz2= np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y-dy), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)
            for sn in self.target:
                # 绘制目标坐标点
                self.ax.scatter(sn.x, sn.y, sn.z,c='red')
        
        for uav in self.uavs:
            # 绘制无人机坐标点
            self.ax.scatter(uav.x, uav.y, uav.z,c='blue')
    """

    def render(self, flag=0):
        # 绘制封闭立方体
        # 参数
        # x,y,z立方体中心坐标
        # dx,dy,dz 立方体长宽高半长
        # fig = plt.figure()

        if flag == 1:
            # 第一次渲染，需要渲染建筑
            z = 0
            # ax = self.fig.add_subplot(1, 1, 1, projection='3d')
            for ob in self.equi:
                # 绘画出所有建筑
                x = ob.x
                y = ob.y
                d = ob.data
                if d > 2700:
                    self.ax.scatter(x, y, c='black')
                else:
                    self.ax.scatter(x, y, c='purple')
            for sn in self.target:
                # 绘制目标坐标点
                self.ax.scatter(sn.x, sn.y, c='red')

        for uav in self.uavs:
            # 绘制无人机坐标点
            self.ax.scatter(uav.x, uav.y, c='blue')

            # 循环
            for position in uav.listdone:
                # 绿色
                self.ax.scatter(position[0], position[1], c='green')

    def step(self, action,i):

        """环境的主要驱动函数，主逻辑将在该函数中实现。该函数可以按照时间轴，固定时间间隔调用

        参数:
            action (object): an action provided by the agent
            i:i号无人机执行更新动作

        返回值:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """

        # 初始化奖励值
        reward = 0.0
        # 判断结束时刻
        done = False

        # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=0

        # 无人机执行行为,info为是否到达目标点
        reward, done, info = self.uavs[i].update(action)

        # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

        # 更新状态
        next_state = self.uavs[i].state()

        return next_state, reward, done, info

    # TODO x,y,z三个坐标都要random
    def reset(self):

        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立"""

        # 重置画布
        # plt.close()
        # self.fig=plt.figure()
        # self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # plt.clf()
        self.uavs =[]
        self.base = []
        self.equi = []
        # self.bds=[]
        self.map = np.zeros((self.len, self.width))  # 重置传感器
        self.datamap = np.zeros((self.len, self.width))  # 重置传感器数据量地图
        self.abstract_datamap = np.zeros((self.len, self.width))  # 重置传感器数据量抽象地图

        self.data_uprange = 2950
        self.data_downrange = 2450
        self.n_equ = 12

        # 随机生成设备位置
        # 区域1
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(5, 18)         # 原训练 45
                y = random.randint(2, 7)         # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域2
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(18, 31)         # 原训练 45
                y = random.randint(7, 13)         # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域3
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(31, 44)         # 原训练 45
                y = random.randint(13, 19)         # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange) + 500
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域4
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(44, 57)         # 原训练 45
                y = random.randint(7, 13)         # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域5
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(57, 70)         # 原训练 45
                y = random.randint(1, 7)         # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 生成固定终点位置
        x = 75         # 原训练45
        y = 5         # 原训练25
        self.target = [destination(x, y)]
        self.map[x, y] = 3
        self.abstract_datamap[x, y] = 4

        # 生成固定无人机位置
        for i in range(self.n_uav):
            # 初始化无人机位置
            x = 2
            # 从地图边缘开始飞行
            y = 2           # 原训练 25
            # 放入列表
            self.uavs.append(UAV(x,y,self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

        # 更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

# TODO 地图正方形
    """
    def reset_test(self):
        # 环境重组测试
        self.uavs = []
        self.base = []
        self.equi = []
        self.map = np.zeros((self.len, self.width))  # 重置障碍物
        # self.WindField=[]
        # 生成随机风力和风向
        # self.WindField.append(np.random.normal(40,5))
        # self.WindField.append(2*math.pi*random.random())


        # 随机生成设备位置
        for i in range(self.n_equi):
            x = 0
            y = 0
            while (1):
                x = random.randint(1, 45)
                y = random.randint(1, 45)
                if self.map[x, y] != 2:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y))
            # 更新地图
            self.map[x, y] = 2

        # 生成固定终点位置
        x = 45
        y = 25
        self.target = [sn(x, y)]
        self.map[x, y] = 3

        # 生成固定无人机位置
        for i in range(self.n_uav):
            # 初始化无人机位置
            x = 0
            # 从地图边缘开始飞行
            y = 25
            # 放入列表
            self.uavs.append(UAV(x, y, self))


        # 更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state
"""

    def reset_test(self):

        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立"""

        # 重置画布
        # plt.close()
        # self.fig=plt.figure()
        # self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # plt.clf()
        self.uavs=[]
        self.base = []
        self.equi = []
        # self.bds=[]
        self.map = np.zeros((self.len, self.width))  # 重置障碍物
        self.datamap = np.zeros((self.len, self.width))  # 重置传感器数据量地图
        self.abstract_datamap = np.zeros((self.len, self.width))  # 重置传感器数据量抽象地图

        self.data_uprange = 2950
        self.data_downrange = 2450
        self.n_equ = 12
        # 随机生成设备位置
        # 区域1
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(5, 15)  # 原训练 45
                y = random.randint(3, 7)  # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域2
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(18, 28)  # 原训练 45
                y = random.randint(7, 13)  # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域3
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(31, 44)  # 原训练 45
                y = random.randint(13, 19)  # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域4
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(48, 57)  # 原训练 45
                y = random.randint(7, 13)  # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        # 区域5
        for i in range(self.n_equ):
            x = 0
            y = 0
            randata = 0
            while (1):
                x = random.randint(60, 70)  # 原训练 45
                y = random.randint(3, 7)  # 原训练 45
                randata = random.randint(self.data_downrange, self.data_uprange)
                if self.map[x, y] != 1:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y, randata))
            # 更新地图
            self.map[x, y] = 1
            self.datamap[x, y] = randata
            if randata > 2500:
                self.abstract_datamap[x, y] = 3
            else:
                self.abstract_datamap[x, y] = 2

        """
        # 随机生成无人机位置
        for i in range(self.n_uav):
            x = 0
            y = 0
            while(1):
                x=random.randint(15,30)
                y=random.randint(10,90)
                if self.map[x,y,z]==0:
                    # 随机生成在无障碍区域
                    break
        """
        # TODO 添加的终点位置生成
        # 生成固定终点位置
        x = 75         # 原训练45
        y = 5         # 原训练25
        self.target = [destination(x, y)]
        self.map[x, y] = 3
        self.abstract_datamap[x, y] = 4

        # 生成固定无人机位置
        for i in range(self.n_uav):
            # 初始化无人机位置
            x = 2
            # 从地图边缘开始飞行
            y = 2           # 原训练 25
            # 放入列表
            self.uavs.append(UAV(x, y, self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

        # 更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state


"""
# TODO 长方形版本
    def reset_test(self):
        # 环境重组测试
        self.uavs = []
        self.base = []
        self.equi = []
        self.map = np.zeros((self.len, self.width))  # 重置障碍物
        # self.WindField=[]
        # 生成随机风力和风向
        # self.WindField.append(np.random.normal(40,5))
        # self.WindField.append(2*math.pi*random.random())

        # 随机生成设备位置
        for i in range(100):
            x = 0
            y = 0
            while (1):
                x = random.randint(1, 75)
                y = random.randint(1, 19)
                if self.map[x, y] != 2:
                    # 随机生成在不同区域
                    break
            # 重新生成设备的位置
            self.equi.append(sn(x, y))
            # 更新地图
            self.map[x, y] = 2

        # 生成固定终点位置
        x = 75
        y = 10
        self.target = [sn(x, y)]
        self.map[x, y] = 3

        # 生成固定无人机位置
        for i in range(self.n_uav):
            # 初始化无人机位置
            x = 0
            # 从地图边缘开始飞行
            y = 5
            # 放入列表
            self.uavs.append(UAV(x, y, self))


        # 更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state
"""
if __name__ == "__main__":
    env=Env()
  
    env.reset()
    env.render()
    plt.pause(30)

