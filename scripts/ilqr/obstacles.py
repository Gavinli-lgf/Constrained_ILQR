import numpy as np 
import math
import pdb

class Obstacle:
    def __init__(self, args, track_id, bb):
        self.args = args
        self.car_length = bb[0]
        self.car_width = bb[1]
        self.track_id = track_id

    """
	输入:npc_traj:npc在控制域horizon内的状态[:, i:i+self.args.horizon]; i:当前时刻在horizon内的序号; 
        ego_state:自车nominal trajectory在i时刻的状态
    输出: b_dot_obs, b_ddot_obs: 代价函数l在i点对npc的一阶和二阶偏导
    (npc障碍物的约束,通过barrier function加入代价函数中.分别对npc的barrier function计算一,二阶导)
	"""
    def get_obstacle_cost_derivatives(self, npc_traj, i, ego_state):

        a = self.car_length + np.abs(npc_traj[2, i]*math.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.ego_rad
        b = self.car_width + np.abs(npc_traj[2, i]*math.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.ego_rad
        
        # 椭圆矩阵形式的的系数矩阵P1
        P1 = np.diag([1/a**2, 1/b**2, 0, 0])

        theta = npc_traj[3, i]
        theta_ego = ego_state[3]

        # 4*4,乘以状态变量X(x,y,v,theta),可以对其中(x,y)部分向量顺时针旋转;(v,theta)部分不变
        transformation_matrix = np.array([[ math.cos(theta), math.sin(theta), 0, 0],
                                          [-math.sin(theta), math.cos(theta), 0, 0],
                                          [               0,               0, 0, 0],
                                          [               0,               0, 0, 0]])
        
        # p_front = [p_x + cosθ*Δl, p_y + sinθ*Δl],推导过程见论文
        ego_front = ego_state + np.array([math.cos(theta_ego)*self.args.ego_lf, math.sin(theta_ego)*self.args.ego_lf, 0, 0])
        # 用于将自车前部状态(主要是坐标)转换到 NPC 局部坐标系下
        diff = (transformation_matrix @ (ego_front - npc_traj[:, i])).reshape(-1, 1) # (x- xo)
        # 构造NPC局部坐标系下的NPC椭圆约束函数c,以及其一阶导数c_dot
        c = 1 - diff.T @ P1 @ diff # Transform into a constraint function
        c_dot = -2 * P1 @ diff
        # 计算自车前部在NPC barrier function中的值b, 一阶导b_dot, 二阶导b_ddot
        b_f, b_dot_f, b_ddot_f = self.barrier_function(self.args.q1_front, self.args.q2_front, c, c_dot)

        # 同上: 计算自车后轴中心在NPC barrier function中的值b, 一阶导b_dot, 二阶导b_ddot
        ego_rear = ego_state - np.array([math.cos(theta_ego)*self.args.ego_lr, math.sin(theta_ego)*self.args.ego_lr, 0, 0])
        diff = (transformation_matrix @ (ego_rear - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_r, b_dot_r, b_ddot_r = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

        # 将自车前,后部在NPC barrier function中的一阶导b_dot, 二阶导b_ddot累加,得到总的NPC约束代价函数
        return b_dot_f + b_dot_r, b_ddot_f + b_ddot_r

    def get_obstacle_cost(self, npc_traj, i, ego_state_nominal, ego_state):
        a = self.car_length + np.abs(npc_traj[2, i]*math.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.ego_rad
        b = self.car_width + np.abs(npc_traj[2, i]*math.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.ego_rad
        
        P1 = np.diag([1/a**2, 1/b**2, 0, 0])

        theta = npc_traj[3, i]
        theta_ego = ego_state[3]
        theta_ego_nominal = ego_state_nominal[3]


        transformation_matrix = np.array([[ math.cos(theta), math.sin(theta), 0, 0],
                                          [-math.sin(theta), math.cos(theta), 0, 0],
                                          [               0,               0, 0, 0],
                                          [               0,               0, 0, 0]])
        
        # front circle
        ego_front_nominal = ego_state_nominal + np.array([math.cos(theta_ego)*self.args.ego_lf, math.sin(theta_ego)*self.args.ego_lf, 0, 0])
        ego_front = ego_state + np.array([math.cos(theta_ego_nominal)*self.args.ego_lf, math.sin(theta_ego_nominal)*self.args.ego_lf, 0, 0])

        x_del = ego_front - ego_front_nominal

        diff = (transformation_matrix @ (ego_front_nominal - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_f, b_dot_f, b_ddot_f = self.barrier_function(self.args.q1_front, self.args.q2_front, c, c_dot)

        cost = b_f + x_del.T @ b_dot_f + x_del.T @ b_ddot_f @ x_del  

        # rear circle
        ego_rear_nominal = ego_state_nominal - np.array([math.cos(theta_ego)*self.args.ego_lr, math.sin(theta_ego)*self.args.ego_lr, 0, 0])
        ego_rear = ego_state - np.array([math.cos(theta_ego_nominal)*self.args.ego_lr, math.sin(theta_ego_nominal)*self.args.ego_lr, 0, 0])

        x_del = ego_rear - ego_rear_nominal

        diff = (transformation_matrix @ (ego_rear_normalized - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_r, b_dot_r, b_ddot_r = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

        cost += b_r + x_del.T @ b_dot_r + x_del.T @ b_ddot_r @ x_del  

        return cost

    """
    输入: q1,q2: 指数型barrier function的两个参数; c: 原始的不等式约束c<0; c_dot: 不等式约束函数的一阶导数
    输出: barrier function的值b, 一阶导b_dot, 二阶导b_ddot
    """
    def barrier_function(self, q1, q2, c, c_dot):
        b = q1*np.exp(q2*c)
        b_dot = q1*q2*np.exp(q2*c)*c_dot
        b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)

        return b, b_dot, b_ddot
