######################################################################
# UAV Class
#UAV 类描述，对无人机的状态参数进行初始化，
# 包括坐标、目标队列、环境、电量、方向、基础能耗、当前能耗、已经消耗能量、
# 侦测半径、周围障碍情况、坠毁概率、距离目标距离、已走步长等。
# 成员函数能返回自身状态，并根据决策行为对自身状态进行更新。
#----------------------------------------------------------------
# UAV class description, initialize the state parameters of the UAV, 
# including coordinates, target queue, environment, power, direction, 
# basic energy consumption, current energy consumption, consumed energy, 
# detection radius, surrounding obstacles, crash probability, distance Target distance, steps taken, etc. 
# Member functions can return their own state and update their own state according to the decision-making behavior.
#################################################################
import math


import numpy as np


class UAV():
    def __init__(self,x,y,ev):

        # 初始化无人机坐标位置
        self.x = x
        self.y = y
        # self.z=z

        # 初始化无人机目标坐标
        # self.target=[ev.target[0].x,ev.target[0].y,ev.target[0].z]

        # 初始化无人机目标坐标
        self.target = [ev.target[0].x, ev.target[0].y]

        # 横、纵坐标
        self.listdone = []

        # 无人机所处环境
        self.ev = ev

        # 定义规划空间大小
        self.len = 80
        self.width = 20

        # 初始化无人机运动情况

        self.bt = 5000
        self.dir = 0   # 无人机水平运动方向，八种情况(弧度)
        self.p_bt = 10   # 无人机基础能耗，能耗/步
        self.coll_bt = 20  # 无人机收集数据能耗
        self.now_bt = 4   # 无人机当前状态能耗
        self.cost = 0   # 无人机已经消耗能量
        self.detect_r = 5  # 无人机探测圆形范围 （格）
        self.collect_done = 0  # 无人机完成收集数据的传感器个数
        self.data_sum = 0  # 无人机收集的总数据量
        self.collect_time = 0  # 无人机收集数据花费的时间

        # 无人机邻近栅格设备情况
        self.ob_space = np.zeros(80)

        # 建立字典
        self.dist = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (1, 1), 5: (-1, 1), 6: (-1, -1), 7: (1, -1), 8: (2, 0), 9: (0, 2),
                     10: (-2, 0), 11: (0, -2), 12: (2, 1), 13: (1, 2), 14: (-1, 2), 15: (-2, 1), 16: (-2, -1), 17: (-1, -2), 18: (1, -2), 19: (2, -1),
                     20: (2, 2), 21: (-2, 2), 22: (-2, -2), 23: (2, -2), 24: (3, 0), 25: (0, 3), 26: (-3, 0), 27: (0, -3), 28: (2, 3), 29: (1, 3),
                     30: (3, 1), 31: (-1, 3), 32: (-2, 3), 33: (4, 0), 34: (-3, 2), 35: (-3, 1), 36: (3, 2), 37: (-3, -1), 38: (-3, -2), 39: (0, 4),
                     40: (-2, -3), 41: (-1, -3), 42: (-4, 0), 43: (1, -3), 44: (2, -3), 45: (0, -4), 46: (3, -2), 47: (3, -1), 48: (-3, 3), 49: (4, 1),
                     50: (4, 2), 51: (4, 3), 52: (3, 4), 53: (2, 4), 54: (1, 4), 55: (-3, -3), 56: (-1, 4), 57: (-2, 4), 58: (-3, 4), 59: (-4, 3),
                     60: (-4, 2), 61: (-4, 1), 62: (3, 3), 63: (-4, -1), 64: (-4, -2), 65: (-4, -3), 66: (-3, -4), 67: (-2, -4), 68: (-1, -4), 69: (3, -3),
                     70: (1, -4), 71: (2, -4), 72: (3, -4), 73: (4, -3), 74: (4, -2), 75: (4, -1), 76: (5, 0), 77: (0, 5), 78: (-5, 0), 79: (0, -5), }

        # 无人机邻近栅格设备情况
        # self.ob_space = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        self.nearest_distance = 5  # 最近设备距离
        self.dir_ob = None     # 距离无人机最近设备相对于无人机的方位
        self.p_crash = 0   # 无人机坠毁概率
        self.done = False   # 终止状态

        # self.distance = abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])   # 无人机当前距离目标点曼哈顿距离
        # self.d_origin = abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])   # 无人机初始状态距离终点的曼哈顿距离

        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1])   # 无人机当前距离目标点曼哈顿距离
        self.d_origin = abs(self.x - self.target[0]) + abs(self.y - self.target[1])   # 无人机初始状态距离终点的曼哈顿距离
        self.step = 0           # 无人机已走步数

    def move(self, num):

        # 利用动作值计算运动改变量
        # TODO 改成5个飞行动作，加上收集动作合计动作空间为6
        collect_flag = 0
        if num == 1:
            return [1, 0, collect_flag]
        elif num == 2:
            return [1, 1, collect_flag]
        elif num == 3:
            return [1, -1, collect_flag]
        elif num == 4:
            return [0, 1, collect_flag]
        # elif num==4:
        # return [0, -1]
        else:
            # 出现其余动作值的情况时，表明程序运行过程中出现错误并抛出异常
            raise NotImplementedError

    def getpos(self, nn, BS, RR):
            Q = np.eye(nn - 1)
            K1 = 0
            K = np.zeros(nn - 1)
            for i in range(nn - 1):
                K[i] = BS[0, i + 1] ** 2 + BS[1, i + 1] ** 2 + BS[2, i + 1] ** 2

            Ga = np.zeros((nn - 1, 4))
            for i in range(nn - 1):
                Ga[i, 0] = -BS[0, i + 1]
                Ga[i, 1] = -BS[1, i + 1]
                Ga[i, 2] = -BS[2, i + 1]
                Ga[i, 3] = -RR[i]

            h = np.zeros(nn - 1)
            for i in range(nn - 1):
                h[i] = 0.5 * (RR[i] ** 2 - K[i] + K1)

            Za0 = np.linalg.pinv(Ga.T @ np.linalg.pinv(Q) @ Ga) @ Ga.T @ np.linalg.pinv(Q) @ h.T

            B = np.eye(nn - 1)
            for i in range(nn - 1):
                B[i, i] = np.sqrt(
                    (BS[0, i + 1] - Za0[0]) ** 2 + (BS[1, i + 1] - Za0[1]) ** 2 + (BS[2, i + 1] - Za0[2]) ** 2)
            FI = B @ Q @ B.T

            Za1 = np.linalg.pinv(Ga.T @ np.linalg.pinv(FI) @ Ga) @ Ga.T @ np.linalg.pinv(FI) @ h.T

            if Za1[3] < 0:
                Za1[3] = abs(Za1[3])

            CovZa = np.linalg.pinv(Ga.T @ np.linalg.pinv(FI) @ Ga)
            sB = np.eye(4)
            for i in range(4):
                sB[i, i] = Za1[i]
            sFI = 4 * sB @ CovZa @ sB
            sGa = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            sh = np.array([Za1[0] ** 2, Za1[1] ** 2, Za1[2] ** 2, Za1[3] ** 2]).T

            Za2 = np.linalg.pinv(sGa.T @ np.linalg.pinv(sFI) @ sGa) @ sGa.T @ np.linalg.pinv(sFI) @ sh
            Za = np.sqrt(np.abs(Za2))

            out = np.abs(Za)

            return out

    def cal_var(self, pos_uav, pos_iot, pos_bs):
        B = 180000  # 带宽
        Ts = 2 / 3 * 10 ** -4
        N0 = 7.08 * 10 ** -15
        belta = 8082.57  # 自由空间损耗
        alpha = 2
        p_tbs = 10
        p_uav = 10
        Nc = 16  # OFDM子载波数量
        pc = 5e-5  # OFDM载波总功率

        # 计算sigma_uav
        snr = (pc / N0) / B
        sump = 0
        for i in range(1, Nc + 1):
            sump = pc / Nc * i ** 2 + sump
        L = Ts ** 2 / (Nc * 8 * np.pi ** 2 * snr * sump)
        distan = np.sum((pos_uav - pos_iot) ** 2)
        sigma_uav = (L * N0 * belta * distan ** (alpha / 2)) / p_uav

        # 假定G2G信道同样也为LOS，求解simga_bs
        sigma_bs = np.zeros(3)
        distan1 = np.sum((pos_bs[0] - pos_iot) ** 2)
        distan2 = np.sum((pos_bs[1] - pos_iot) ** 2)
        distan3 = np.sum((pos_bs[2] - pos_iot) ** 2)

        sigma_bs[0] = (L * N0 * belta * distan1 ** (alpha / 2)) / p_tbs
        sigma_bs[1] = (L * N0 * belta * distan2 ** (alpha / 2)) / p_tbs
        sigma_bs[2] = (L * N0 * belta * distan3 ** (alpha / 2)) / p_tbs

        var = np.zeros((3, 3))
        var[0, 0] = sigma_bs[0] + sigma_bs[1]
        var[0, 1] = sigma_bs[0]
        var[0, 2] = sigma_bs[0]
        var[1, 0] = sigma_bs[0]
        var[1, 1] = sigma_bs[0] + sigma_bs[2]
        var[1, 2] = sigma_bs[0]
        var[2, 0] = sigma_bs[0]
        var[2, 1] = sigma_bs[0]
        var[2, 2] = sigma_bs[0] + sigma_uav

        return var

    def cal_crlb(self, var, pos_uav, pos_iot, dis, pos_bs):
        c = 3e8

        # calculate Jacobian matrix
        H = np.zeros((3, 3))

        H[0, 0] = (pos_iot[0] - pos_bs[0, 0]) / dis[0] - (pos_iot[0] - pos_bs[1, 0]) / dis[1]
        H[0, 1] = (pos_iot[1] - pos_bs[0, 1]) / dis[0] - (pos_iot[1] - pos_bs[1, 1]) / dis[1]
        H[0, 2] = (pos_iot[2] - pos_bs[0, 2]) / dis[0] - (pos_iot[2] - pos_bs[1, 2]) / dis[1]

        H[1, 0] = (pos_iot[0] - pos_bs[0, 0]) / dis[0] - (pos_iot[0] - pos_bs[2, 0]) / dis[2]
        H[1, 1] = (pos_iot[1] - pos_bs[0, 1]) / dis[0] - (pos_iot[1] - pos_bs[2, 1]) / dis[2]
        H[1, 2] = (pos_iot[2] - pos_bs[0, 2]) / dis[0] - (pos_iot[2] - pos_bs[2, 2]) / dis[2]

        H[2, 0] = (pos_iot[0] - pos_bs[0, 0]) / dis[0] - (pos_iot[0] - pos_uav[0]) / dis[3]
        H[2, 1] = (pos_iot[1] - pos_bs[0, 1]) / dis[0] - (pos_iot[1] - pos_uav[1]) / dis[3]
        H[2, 2] = (pos_iot[2] - pos_bs[0, 2]) / dis[0] - (pos_iot[2] - pos_uav[2]) / dis[3]

        CRLB = np.trace(np.linalg.inv(H.T @ np.linalg.inv(var) @ H)) * c * c

        return CRLB

    def cal_dataamount(self):
        theta = 7.08 * 10 ** -5  # 噪声功率
        belta = 8082.57  # 自由空间损耗率
        alpha = 2  # 指数
        pt = 10 ** -5  # 信号功率
        Bw = 180000 / 16  # 带宽
        sum = 0  # 总信息量

        for i in range(0, 28):
            if self.ob_space[i] > 2:
                [disx, disy] = self.dist[i]
                dd = (3600 + (disy * 20) ** 2 + (disx * 20) ** 2)
                fade = belta / dd ** (alpha / 2)
                sinr = fade * pt / (fade * pt + theta)
                c = Bw * math.log2(1 + sinr)
                sum = c + sum

        return sum

    # 设置函数用于更新传感器数据量
    def collect(self, num):
        theta = 7.08 * 10 ** -5  # 噪声功率
        belta = 8082.57   # 自由空间损耗率
        alpha = 2   # 指数
        pt = 10 ** -5  # 信号功率
        Bw = 180000/16  # 带宽
        collect_flag = 0 # 成功执行完下述代码，表示成功收集数据
        data_amount = 0
        dx = 0
        dy = 0
        datasum = 0

        for i in range(0, 48):
            if self.ob_space[i] > 0:
                [disx, disy] = self.dist[i]
                dd = (3600 + (disy * 20) ** 2 + (disx * 20) ** 2)
                fade = belta / dd ** (alpha / 2)
                sinr = fade * pt / (fade * pt + theta)
                c = Bw * math.log2(1 + sinr)
                data_amount = c * 1.25

                if self.x + disx < 0 or self.x + disx >= self.len or self.y + disy < 0 or self.y + disy >= self.width:
                    collect_flag = 0
                    # 在地图范围内则要更新
                else:
                    self.ev.datamap[self.x + disx, self.y + disy] = self.ev.datamap[self.x + disx, self.y + disy] - c
                    collect_flag += 1
                    self.collect_time = self.collect_time + 1

                tip = 2950
                tip_thres = 2500
                # 如果设备不在地图范围内
                if self.x + disx < 0 or self.x + disx >= self.len or self.y + disy < 0 or self.y + disy >= self.width:
                    tip = 2950
                # 在地图范围内则要更新
                else:
                    if tip >= self.ev.datamap[self.x + disx, self.y + disy] > tip_thres:
                        self.ev.abstract_datamap[self.x + disx, self.y + disy] = 3
                        datasum = datasum + data_amount
                    elif tip_thres >= self.ev.datamap[self.x + disx, self.y + disy] > 0:
                        self.ev.abstract_datamap[self.x + disx, self.y + disy] = 2
                        datasum = datasum + data_amount
                    else:
                        self.ev.abstract_datamap[self.x + disx, self.y + disy] = 0
                        self.ev.map[self.x + disx, self.y + disy] = 0
                        self.collect_done = 1 + self.collect_done
                        datasum = datasum + data_amount + self.ev.datamap[self.x + disx, self.y + disy]
                        self.ev.datamap[self.x + disx, self.y + disy] = 0
                        self.listdone.append((self.x + disx, self.y + disy))

        if collect_flag == 0:
            dx = 1
            dy = 0

        return [datasum, collect_flag, dx, dy]

    def state(self):

        # 计算目标位置和当前位置在三个坐标上的差值
        dx = self.target[0]-self.x
        dy = self.target[1]-self.y

        # 利用列表来存储一系列状态变量的值
        state_grid = [self.x, self.y, dx, dy, self.target[0], self.target[1], self.d_origin, self.step, self.distance, self.now_bt, self.cost]

        # 根据字典更新临近栅格状态
        self.ob_space = []

        # TODO 遍历收集数据范围，将数据量等级放入状态中
        for i in range(0, 48):

            # 首先将索引对应的字典值提取出来
            [a, b] = self.dist[i]

            # 然后判断临近栅格是否超出地图边界
            if self.x + a < 0 or self.x + a >= self.len or self.y + b < 0 or self.y + b >= self.width:

                # 设置超出地图边界的区域对应值
                self.ob_space.append(-2)

                # 将该信息加入到状态中
                state_grid.append(-2)

            # 最后考虑位于地图内部的栅格区域
            else:

                # 将无人机临近各个栅格的地图信息导入，其中：认为存在多数据设备的区域设置为 2 ，少数据为1，无传感器设置为0，终点设置为-1
                self.ob_space.append(self.ev.abstract_datamap[self.x + a, self.y + b])

                # 将该信息加入到状态中
                state_grid.append(self.ev.abstract_datamap[self.x + a, self.y + b])

        # 遍历无人机周围所有位置，借助字典来更新状态
        for i in range(48, 80):

            # 首先将索引对应的字典值提取出来
            [a, b] = self.dist[i]

            # 然后判断临近栅格是否超出地图边界
            if self.x + a < 0 or self.x + a >= self.len or self.y + b < 0 or self.y + b >= self.width:

                # 设置超出地图边界的区域对应值
                self.ob_space.append(-2)

                # 将该信息加入到状态中
                state_grid.append(-2)

            # 最后考虑位于地图内部的栅格区域
            else:

                # 将无人机临近各个栅格的地图信息导入，其中：认为存在设备的区域设置为 1 ， 其余区域设置为 0
                self.ob_space.append(self.ev.map[self.x + a, self.y + b])

                # 将该信息加入到状态中
                state_grid.append(self.ev.map[self.x + a, self.y + b])

        return state_grid

    def update(self, action):

        # 更新无人机状态
        dx,dy = [0, 0]
        temp = action

        # 相关参数

        # 撞毁参数
        b = 3
        # 目标参数
        wt = 0.005
        # 爬升参数
        wc = 0.07
        # 定位参数
        wpos = 1
        # 能量损耗参数
        we = 0
        # 数据收集参数（原0.0200）
        wdata = 1.0000
        # 风阻能耗参数
        c = 0.05
        # 坠毁概率惩罚增益倍数
        crash = 0
        # 距离终点的距离变化量
        Ddistance = 0
        # 用于定位的总基站(包括无人机)的数量
        BSN = 4
        # 定义基站的位置
        BS = np.array([[0, 40, 80, self.x], [20, 0, 20, self.y], [10, 20, 30, 60]])
        # 定位算法中的误差
        Noise = 0.01
        # 定义基站的位置
        pos_bs = BS[:, 1:].T
        # 定义无人机的位置
        pos_uav = BS[:, 0]
        # 定义定位精度的克拉美罗界限
        CRLB = 0
        # 定义本轮数据收集量
        data_amount = 0
        # 定义CRLB阈值
        threshold_c = 1
        # 定义定位惩罚项
        r_pos = 0

        # 计算无人机坐标变更值
        if temp % 5 > 0:
            [dx, dy, collect_flag] = self.move(temp % 5)
        else:
            [data_amount, collect_flag, dx, dy] = self.collect(int(temp % 5))

        # 如果无人机静止不动，给予大量惩罚
        if dx == 0 and dy == 0 and collect_flag == 0:
            return -10000, False, False

        # 更新坐标
        self.x = self.x + dx
        self.y = self.y + dy

        # 正代表接近目标，负代表远离目标
        Ddistance = self.distance - (abs(self.x - self.target[0]) + abs(self.y - self.target[1]))

        # 更新距离值
        # self.distance=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])
        # self.step+=abs(dx)+abs(dy)+abs(dz)

        # 更新距离值
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1])
        self.step += abs(dx) + abs(dy)

        flag = 1
        if abs(dy) == dy:
            flag = 1
        else:
            flag = -1

        # 计算能耗与相关奖励
        # 更新当前能耗状态
        self.now_bt = self.p_bt * math.sqrt((dx ** 2 + dy ** 2)) + self.coll_bt * collect_flag

        # 更新能量损耗状态
        self.cost = self.cost + self.now_bt

# TODO 新添加的CRLB计算
        # 计算总的CRLB
        for i in range(0, 80):
            if self.ob_space[i]==2:
                # 根据索引i查询距离
                [disx, disy] = self.dist[i]
                R0 = np.zeros(BSN)
                R = np.zeros(BSN - 1)
                for j in range(BSN):
                    R0[j] = np.sqrt((BS[0, j] - disx) ** 2 + (BS[1, j] - disy) ** 2 + (BS[2, j]) ** 2)
                for k in range(BSN - 1):
                    R[k] = R0[k + 1] - R0[0] + Noise * np.random.randn()
                X = self.getpos(BSN, BS, R)
                var = self.cal_var(pos_uav, X.T, pos_bs)
                d_equiandbs = np.zeros(BSN)
                for i in range(3):
                    d_equiandbs[i] = np.sqrt(np.sum((pos_bs[i] - X) ** 2))
                d_equiandbs[3] = np.sqrt(np.sum((pos_uav - X) ** 2))
                CRLB = self.cal_crlb(var, pos_uav, X.T, d_equiandbs, pos_bs) + CRLB

        # 计算CRLB奖励
        # r_pos = 0.01 * wpos / CRLB
        if CRLB > threshold_c:
            r_pos = -40


        # 计算数据收集奖励
        r_data = data_amount * wdata
        self.data_sum = self.data_sum + data_amount

        # 计算总奖励
        # r = r_pos + r_data
        # r = r_data

        r = r_data + r_pos

        # 惩罚项添加
        if self.cost > self.bt:
            # 电量耗尽，给予大量惩罚
            return r-80000, True, 3

        # 从5改为8
        if self.distance <= 5:
            # 到达目标点，给予f大量奖励
            # 同时给予能耗奖励，走的步数越少越好
            # 同时给予完成度奖励，收集越多奖励越大
            return r + 25000 + 0 * self.collect_done, True, 1

            # 如果无人机飞出边界，给予大量惩罚
        if self.x < 0 or self.x > 80 or self.y < 0 or self.y > 20:
            return -81000, True, 2

        # 建议添加定位精度的约束项，当所有目标完成定位后，结束

        return r, False, 4




