import argparse
import logging
import os
import platform
import pdb
import math
import sys
import time

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patches
from sklearn.preprocessing import normalize

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    print("Cannot add the common path {}".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(sys.path)
from arguments import add_arguments
from PolyRect import PolyRect
from ilqr.iLQR import iLQR

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)

PI = math.pi
colors = ['r', 'g', 'b', 'k']

# Lanes defined at 4, 0, -4

class PySimulator:

    """
    初始化仿真器，设置仿真参数、车辆状态和绘图参数。
    创建 iLQR 控制器，并模拟 NPC 车辆的状态。
    初始化车辆的图形表示，并将其添加到绘图中。
    """
    def __init__(self, args, SimParams, NPC_start, NPC_control):
        self.args = args
        # num_vehicles is only 2 for now
        self.NPC_dict = {}
        self.patches = []

        self.simparams = SimParams
        self.num_vehicles = self.simparams.num_vehicles
        self.navigation_agent = None    # iLQR控制器对象
        self.current_ego_state = self.simparams.start_state # 记录自车的当前状态
        # self.last_ego_states = [self.simparams.start_state[0:2]]
        self.last_ego_states = np.array([self.simparams.start_state[0], self.simparams.start_state[1]])

        # Plot parameters
        self.fig = plt.figure(figsize=(25, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.set_xlim(-5, self.simparams.map_lengthx+5)
        self.ax.set_ylim(-self.simparams.map_lengthy, self.simparams.map_lengthy)
        self.ax.axhline(y=self.simparams.lane1, c='k', lw='4')
        self.ax.axhline(y=self.simparams.lane2, c='k', lw='2', ls='--')
        self.ax.axhline(y=self.simparams.lane3, c='k', lw='4')

        # Plot Local Plan
        self.x_local_plan = []
        self.y_local_plan = []

        self.x_desired_plan = []
        self.y_desired_plan = []

        self.local_plan_plot, = plt.plot([], [], 'go', ms=10)
        self.desired_plan_plot, = plt.plot([], [], 'co', ms=5)
        self.last_states_plots, = plt.plot([], [], 'mo', ms=5)
        trajs = [self.local_plan_plot, self.desired_plan_plot, self.last_states_plots,] # C3
        self.line_plots = list(trajs)

        self.create_ilqr_agent()
        self.NPC_states = []    # 存储非玩家角色(NPC)车辆的状态
        self.simulate_npc(NPC_start, NPC_control)
        self.count = 0

        # Ego vehicle is first
        for i in range(num_vehicles):
            self.NPC_dict[i] = PolyRect(self.simparams.car_dims)
            self.patches.append(self.NPC_dict[i].getPatch(self.ax, colors[i]))
            self.ax.add_patch(self.patches[i]) # C1
        
        # May need to add loop here
        # self.ax.add_patch(self.patches[0]) C2
        # self.ax.add_patch(self.patches[1])

        self.all_patches = self.patches + self.line_plots


    """
    模拟 NPC 车辆在初始状态 init_state 下，经过输入序列 control 后的状态变化，并将其存储在 NPC_states 中。
    """
    def simulate_npc(self, init_state, control):
        self.NPC_states.append(init_state)
        # 将控制输入矩阵 control 扩展，添加零值列以匹配仿真时间步长 self.args.horizon(默认40)
        control = np.hstack((control, np.zeros((2, self.args.horizon))))
        # 遍历控制输入矩阵的每一列（每个时间步的控制输入）。
        for i in range(control.shape[1]):
            # 调用 run_model_simulation 方法，给定当前状态和控制输入，计算下一个状态 NPC_next_state。
            NPC_next_state = self.run_model_simulation(self.NPC_states[i], control[:, i])
            # 将计算得到的下一个状态 NPC_next_state 添加到 NPC_states 列表中。
            self.NPC_states.append(NPC_next_state)
        self.NPC_states = np.array(self.NPC_states).T   # 从141*4，转置为4*141(注：list没有shape，单array有)
        

    """
    自车参考轨迹的位置(x,y)数组，大小(2 * map_lengthx+20)，并在绘图中显示。
    """
    def create_global_plan(self):
        y = self.simparams.desired_y
        self.plan_ilqr = [] # 自车参考轨迹的位置(x,desired_y)数组，大小(2 * map_lengthx+20)
        for i in range(0, self.simparams.map_lengthx+20):
            self.plan_ilqr.append(np.array([i, y]))
        self.plan_ilqr = np.array(self.plan_ilqr)
        self.ax.axhline(y=y, c='r', lw='2')

    # def init_sim(self):
    #     return self.patches[0], self.patches[1], self.local_plan_plot, self.desired_plan_plot,
    def init_sim(self): #C4
        return self.all_patches

    # 将自车当前状态转换为如下"5*3"的array：[[x, y, 0], [v, 0, 0], [0, 0, theta], [0, 0, 0], [0, 0, 0]]
    def get_ego_states(self):
        ego_states = np.array([[self.current_ego_state[0], self.current_ego_state[1],                         0],
                               [self.current_ego_state[2],                         0,                         0],
                               [                        0,                         0, self.current_ego_state[3]],
                               [                        0,                         0,                         0],
                               [                        0,                         0,                         0]])
        return ego_states

    # 返回NPC车辆的长宽array,[4, 2]
    def get_npc_bounding_box(self):
        return self.simparams.car_dims

    # 返回NPC车辆的状态，从第i个时间步开始的self.args.horizon个时间步
    def get_npc_states(self, i):
        return self.NPC_states[:, i:i+self.args.horizon]

    # 创建 iLQR 控制器对象，并设置自车参考轨迹self.plan_ilqr。
    def create_ilqr_agent(self):
        self.create_global_plan()   # 生成自车参考轨迹的位置(x,desired_y)数组，大小(2 * map_lengthx+20)，self.plan_ilqr
        self.navigation_agent = iLQR(self.args, self.get_npc_bounding_box())
        self.navigation_agent.set_global_plan(self.plan_ilqr)

    """
    运行 iLQR 控制器的一个时间步，并返回期望路径、本地路径和控制输入。
    """
    def run_step_ilqr(self):
        assert self.navigation_agent != None, "Navigation Agent not initialized"
        
        # 使用 time.process_time() 记录运行一次run_step需要的时间。
        start = time.process_time()
        desired_path, local_plan, control = self.navigation_agent.run_step(self.get_ego_states(), self.get_npc_states(self.count))
        print(time.process_time() - start)
        
        self.count += 1
        print("Controller: Acc {} Steer: {}".format(control[0, 0], control[1, 0]))

        return desired_path, local_plan, control[:, 0]
 
    def animate(self, i):
        # Get new ego patch
        desired_path, local_plan, control = self.run_step_ilqr()
        self.current_ego_state = self.run_model_simulation(self.current_ego_state, control)

        self.last_ego_states = np.vstack((self.last_ego_states, self.current_ego_state[0:2]))
        
        self.NPC_dict[0].createCuboid([self.current_ego_state[0], self.current_ego_state[1], self.current_ego_state[3]]) # Update ego vehicle patch
        self.patches[0].set_xy(self.NPC_dict[0].getCorners()) # Update ego vehicle patch

        # Get new NPC patch
        # pdb.set_trace()
        for j in range(1,self.num_vehicles):
            self.NPC_dict[j].createCuboid([self.NPC_states[0, i], self.NPC_states[1, i], self.NPC_states[3, i]])
            self.patches[j].set_xy(self.NPC_dict[j].getCorners())

        # Get local plan
        self.x_local_plan = local_plan[:, 0]
        self.y_local_plan = local_plan[:, 1]
        self.local_plan_plot.set_data(self.x_local_plan, self.y_local_plan)
        
        self.last_states_plots.set_data(self.last_ego_states[:, 0],self.last_ego_states[:, 1])

        #Get desired plan
        self.x_desired_plan = desired_path[:, 0]
        self.y_desired_plan = desired_path[:, 1]
        self.desired_plan_plot.set_data(self.x_desired_plan, self.y_desired_plan)

        # return self.patches[0], self.patches[1], self.local_plan_plot, self.desired_plan_plot,
        return self.all_patches

    def run_simulation(self):
        anim = animation.FuncAnimation(self.fig, self.animate,
                               init_func=self.init_sim,
                               frames=self.simparams.sim_time,
                               interval=1000,
                               blit=True,
                               repeat=False)
        plt.show()

    # 给定当前状态state和控制输入control，计算并返回下一个状态next_state
    def run_model_simulation(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control[0] = np.clip(control[0], self.simparams.accel_min, self.simparams.accel_max)
        control[1] = np.clip(control[1], state[2]*tan(self.simparams.steer_min)/self.simparams.wheelbase, state[2]*tan(self.simparams.steer_max)/self.simparams.wheelbase)
        
        Ts = self.simparams.dt
        next_state = np.array([state[0] + cos(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               state[1] + sin(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               np.clip(state[2] + control[0]*Ts, 0.0, self.simparams.max_speed),
                              (state[3] + control[1]*Ts)%(2*np.pi)])  # wrap angles between 0 and 2*pi
        # print("Next state {}".format(next_state))

        return next_state

"""
SimParams:定义仿真中用到的地图、车辆基本参数信息
"""
class SimParams:
    dt = 0.1            #
    sim_time = 100      # 仿真总时间，设置为 100 个时间步，即 10 秒。
    map_lengthx = 50    # 地图的纵向长度s，设置为 50 米
    map_lengthy = 6     # 地图单边宽度l为6m(两侧共12m)
    lane1 = 5           # 单车道lane宽度为5m
    lane2 = 0
    lane3 = -5
    num_vehicles = 2    # 仿真车数量2(包括自车)

    ## Car Parameters
    car_dims = np.array([4, 2])             # ego车长4m，宽2m
    start_state = np.array([5, 0, 0, 0])    # ego的初始状态x0
    max_speed = 180/3.6                     # ego最大车速(m/s)
    wheelbase = 2.94
    steer_min = -1.0    # 方向盘的最小和最大转角，分别设置为 -1.0 和 1.0。
    steer_max = 1.0
    accel_min = -5.5    # 加速度的最小和最大值，分别设置为 -5.5 和 3.0
    accel_max = 3.0
    desired_y = 2.5      # ego期望的横向位置2.5m（纵向位置是0~50）
    NPC_max_acc = 0.75   # NPC 车辆的最大加速度，设置为 0.75




if __name__ == "__main__":
    # 解析命令行参数,并将参数存储在args中
    argparser = argparse.ArgumentParser(description='CARLA CILQR')
    add_arguments(argparser)    # 在arguments.py中定义的参数
    args = argparser.parse_args()

    NPC_start = np.array([10, 2.5, 0, 0])   #obs的初始状态(x,y,v,theta)=(10, 2.5, 0, 0)
    # 存储NPC车辆的控制输入,大小"2*sim_time",第一行是加速度acc，第二行是横摆角速度yawrate；
    NPC_ctrl = np.zeros((2, SimParams.sim_time))
    # acc从NPC_max_acc(0.75)分sim_time(100)个时间步递减到0;yawrate全为0.
    NPC_ctrl[0,:] = np.linspace(SimParams.NPC_max_acc, 0, SimParams.sim_time)
    num_vehicles = 2
    # 创建仿真器pysim，并传入预定义参数
    pysim = PySimulator(args, SimParams, NPC_start, NPC_ctrl)
    pysim.run_simulation()
