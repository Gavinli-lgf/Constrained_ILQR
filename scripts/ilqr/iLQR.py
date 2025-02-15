import math
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pdb
import sys

from ilqr.vehicle_model import Model
from ilqr.local_planner import LocalPlanner
from ilqr.constraints import Constraints


class iLQR():
    """
    args : arguments.py中定义的参数; obstacle_bb : 障碍物的bounding box(其实是一个长宽数组); 
    """
    def __init__(self, args, obstacle_bb, verbose=False):
        self.args = args
        self.Ts = args.timestep
        self.N = args.horizon
        self.tol = args.tol
        self.obstacle_bb = obstacle_bb
        self.verbose = verbose
        
        self.global_plan = None
        self.local_planner = LocalPlanner(args) # 局部规划对象:根据自车位置，从自车参考轨迹上取20个点，并拟合满足5次多项式
        self.vehicle_model = Model(args)
        self.constraints = Constraints(args, obstacle_bb)
        
        # initial nominal trajectory(初始化 self.control_seq 大小(2, 40)，首行都为的0.5数组)
        self.control_seq = np.zeros((self.args.num_ctrls, self.args.horizon)) # 大小(2, 40)
        self.control_seq[0, :] = np.ones((self.args.horizon)) * 0.5 # 默认控制序列中的初始加速度都是0.5
        self.debug_flag = 0

        self.lamb_factor = 10
        self.max_lamb = 1000

        # self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1,3, num=0, figsize=(20, 5))

    
    """
    global_plan:自车参考轨迹的位置(x,desired_y)的list, 转换为array大小为(map_lengthx+20, 2)
    """
    def set_global_plan(self, global_plan):
        self.global_plan = global_plan
        self.local_planner.set_global_planner(self.global_plan)

    """
    输入: X_0:自车当前状态 (x, y, v, yaw);  U:控制序列(2, 40),默认初始a都为0.5;
    输出: X: 车辆运动学方程在初始状态下, 通过控制序列U, 得到车辆的状态序列(4, 41)
    功能: 给定车辆的初始状态与控制序列，通过状态转移方程，计算生成的名义轨迹(nominal trajectory)
    """
    def get_nominal_trajectory(self, X_0, U):
        # 初始化X为(4, 41)大小的zero数组，X[:, 0]为自车当前状态
        X = np.zeros((self.args.num_states, self.args.horizon+1))
        X[:, 0] = X_0
        for i in range(self.args.horizon):
            # 使用车辆运动学方程，根据当前状态和控制输入，计算下一个状态
            X[:, i+1] = self.vehicle_model.forward_simulate(X[:, i], U[:, i])
        return X

    """
    输入: X:自车在 X_0 初始状态下,以nominal输入 U 获得的 nominal trajectory(4,41);  U:自车的nominal输入(2,40); (horizon=40)
        k:前馈序列(2,4,40);  K:反馈增益序列(2,40)
    输出: X_new, U_new: 通过前向滚动再次获取控制点处的新控制值 U_new(2,40) 和新状态 X_new(4,41)
    注: 并未在更新U时使用线性搜索因子α,而是直接使用k和K计算
    """
    def forward_pass(self, X, U, k, K):
        X_new = np.zeros((self.args.num_states, self.args.horizon+1))
        X_new[:, 0] = X[:, 0] # 初始状态不变
        U_new = np.zeros((self.args.num_ctrls, self.args.horizon))
        # Do a forward rollout and get states at all control points
        for i in range(self.args.horizon):
            U_new[:, i] = U[:, i] + k[:, i] + K[:, :, i] @ (X_new[:, i] - X[:, i])
            # 使用车辆运动学方程，根据当前状态和控制输入，计算下一个状态
            X_new[:, i+1] = self.vehicle_model.forward_simulate(X_new[:, i], U_new[:, i])
        return X_new, U_new

    """
    输入:X:自车在 X_0 初始状态下,以nominal输入 U 获得的 nominal trajectory(4,41);  U:自车的nominal输入(2,40); (horizon=40)
        poly_coeff:局部规划拟合多项式的系数;  x_local_plan:局部参考路径的x坐标;  npc_traj:npc在控制域horizon内的状态[:, i:i+self.args.horizon];
        lamb:Regularization parameter;
    输出: k, K: 通过backward pass计算得到的最优控的反馈增益序列 K(2,4,40) ,前馈序列k(2,40)
    """
    def backward_pass(self, X, U, poly_coeff, x_local_plan, npc_traj, lamb):
        # Find control sequence that minimizes Q-value function
        # Get derivatives of Q-function wrt to state and control(代价函数l对x, u的一阶和二阶偏导.即l对horizon内的每个状态和控制量都求了偏导数)
        # 从包含约束barrier function的代价函数l中求:l_x (4, 40), l_xx (4, 4, 40), l_u (2, 40), l_uu (2, 2, 40), l_ux (2, 4, 40)
        l_x, l_xx, l_u, l_uu, l_ux = self.constraints.get_cost_derivatives(X[:, 1:], U, poly_coeff, x_local_plan, npc_traj) 
        # 系统状态转移方程f对x,u的一阶偏导df_dx(4,4,40),df_du(4,2,40).对应原始状态转移方程中的A,B
        df_dx = self.vehicle_model.get_A_matrix(X[2, 1:], X[3, 1:], U[0,:])
        df_du = self.vehicle_model.get_B_matrix(X[3, 1:])
        # Value function at final timestep is known最后一个点的状态价值是已知的,(即只有状态,没有输入. V_xx就是代价函数l对x的二阶偏导数l_xx, V_x是一阶偏导l_x)
        V_x = l_x[:,-1] 
        V_xx = l_xx[:,:,-1]
        # Allocate space for feedforward and feeback term(为反馈增益K,前馈增益k分配空间,K:(2, 4, 40),k:(2, 40))
        k = np.zeros((self.args.num_ctrls, self.args.horizon))
        K = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon))
        # Run a backwards pass from N-1 control step(从最后一个控制步开始，进行反向传播计算)
        for i in range(self.args.horizon-1,-1,-1):
            Q_x = l_x[:,i] + df_dx[:,:,i].T @ V_x
            Q_u = l_u[:,i] + df_du[:,:,i].T @ V_x
            Q_xx = l_xx[:,:,i] + df_dx[:,:,i].T @ V_xx @ df_dx[:,:,i] 
            Q_ux = l_ux[:,:,i] + df_du[:,:,i].T @ V_xx @ df_dx[:,:,i]
            Q_uu = l_uu[:,:,i] + df_du[:,:,i].T @ V_xx @ df_du[:,:,i]
            # Q_uu_inv = np.linalg.pinv(Q_uu)
            # 如下4行通过特征值分解法计算矩阵 Q_uu 的伪逆，并在此过程中确保特征值非负，并增加正则化参数以提高数值稳定性。
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = np.dot(Q_uu_evecs,np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

    
            # Calculate feedforward and feedback terms(计算前馈项 k 和反馈增益 K)
            k[:,i] = -Q_uu_inv @ Q_u
            K[:,:,i] = -Q_uu_inv @ Q_ux
            # Update value function for next time step(计算当前价值函数的 V_x 和 V_xx ,用于下一次迭代)
            V_x = Q_x - K[:,:,i].T @ Q_uu @ k[:,i]
            V_xx = Q_xx - K[:,:,i].T @ Q_uu @ K[:,:,i]
        
        return k, K


    """
    输入: ego_state:自车当前状态,大小(5,3); npc_traj:npc在控制域horizon内的运行状态[:, i:i+self.args.horizon](默认horizon=40)
    输出: traj(11,2):cilqr得到的horizon内的(x,y)轨迹;  ref_traj(2, 20):拟合后的局部参考路径;  
          U(2, 40):cilqr得到的horizon内的最优控制序列;
    """
    def run_step(self, ego_state, npc_traj):
        assert self.global_plan is not None, "Set a global plan in iLQR before starting run_step"

        # 计算拟合后的局部参考路径 ref_traj (2, 20)和拟合多项式的系数 poly_coeff
        self.local_planner.set_ego_state(ego_state)
        ref_traj, poly_coeff = self.local_planner.get_local_plan()

        # X_0:自车当前状态 (x, y, v, yaw)
        X_0 = np.array([ego_state[0][0], ego_state[0][1], ego_state[1][0], ego_state[2][2]])

        # self.control_seq[:, :-1] = self.control_seq[:, 1:]
        # self.control_seq[:, -1] = np.zeros((self.args.num_ctrls))

        # X(4, 41), U(2, 40): 通过iLQR算法得到的 horizon 范围内的最优状态,控制序列
        X, U = self.get_optimal_control_seq(X_0, self.control_seq, poly_coeff, ref_traj[:, 0], npc_traj)
        # 对X序列提取前2个维度(x,y)坐标,并按horizon/10即4的间隔采样，生成traj(11,2)
        traj = X[:2, ::int(self.args.horizon/10)].T 

        self.control_seq = U
        # self.plot(U, X, ref_traj)
        return traj, ref_traj, U #self.filter_control(U,  X[2,:])

    """
    输入: X_0:自车当前状态 (x, y, v, yaw);  U:控制序列(2, 40),默认初始a都为0.5;  poly_coeff:局部参考轨迹拟合多项式的系数; 
         x_local_plan:局部参考轨迹的x坐标;  npc_traj:npc在控制域horizon内的状态[:, i:i+self.args.horizon](默认horizon=40)
    输出: X(4, 41), U(2, 40): 通过iLQR算法得到的 horizon 范围内的最优状态,控制序列
    注: 这里将公式推导中的正定系数 μ 和线性化搜索因子 α 整合为一个参数lamb使用
    """
    def get_optimal_control_seq(self, X_0, U, poly_coeff, x_local_plan, npc_traj):
        # 使用车辆运动学在初始状态 X_0 下, 通过控制序列 U (加速度0.5), 得到车辆的状态序列 X:(4, 41)
        X = self.get_nominal_trajectory(X_0, U)
        J_old = sys.float_info.max  # Initialize cost to a float largest value
        lamb = 1 # Regularization parameter

        # Run iLQR for max iterations(max_iters:20)
        for itr in range(self.args.max_iters):
            # k, K: 通过backward pass计算得到的最优控的反馈增益序列 K(2,40) ,前馈序列k(2,4,40)
            k, K = self.backward_pass(X, U, poly_coeff, x_local_plan, npc_traj, lamb)
            # Get control values at control points and new states again by a forward rollout
            # 通过前向滚动再次获取控制点处的新控制值 U_new(2,40) 和新状态值 X_new(4,41)
            X_new, U_new = self.forward_pass(X, U, k, K)
            # J(标量):X 和 U 与局部参考轨迹(poly_coeff,x_local_plan)的代价. (npc_traj未使用)
            J_new = self.constraints.get_total_cost(X, U, poly_coeff, x_local_plan, npc_traj)
            
            if J_new < J_old:
                # lamb小收敛速度更快,但容易不稳定发散
                X = X_new
                U = U_new
                lamb /= self.lamb_factor
                if (abs(J_old - J_new) < self.args.tol):
                    # 如果J_new<J_old,且,连续2次代价的变化量小于收敛的阈值，则停止迭代
                    print("Tolerance reached")
                    break
            else:
                # 如果J_new>=J_old,则正则化参数lamb扩大10倍(超过1000则停止迭代).(lamb大系统稳定不发散,但收敛更慢)
                lamb *= self.lamb_factor
                if lamb > self.max_lamb:
                    break
            
            J_old = J_new
        # print(J_new)
        return X, U

    def filter_control(self, U, velocity):
        U[1] = np.arctan2(self.args.wheelbase*U[1],velocity[:-1])
        return U

    def plot(self, control, X, ref_traj):
        self.ax1.clear()
        self.ax1.plot(np.arange(len(control[0])), control[0,:], color='g', label='Acc')
        self.ax1.plot(np.arange(len(control[0])), control[1,:], color='b', label='Yaw Rate')
        self.ax1.set_ylabel('Values')
        self.ax1.set_xlabel('Time')
        self.ax1.set_title('Controls',fontsize=18)
        # self.ax1.xlim(0, len(control[0]))
        # self.ax1.ylim(-6, 6)
        # self.ax1.axis('equal')
        self.ax1.legend()
        self.ax1.grid()

        self.ax2.clear()
        self.ax2.plot(ref_traj[:, 0], ref_traj[:, 1], color='r', label='Ref Traj')
        self.ax2.plot(X[0, :], X[1, :], color='g', label='Real Traj')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlabel('x')
        self.ax2.set_title('Position Trajectory',fontsize=18)
        self.ax2.legend()
        self.ax2.grid()
        # plt.legend()
        
        self.ax3.clear()
        self.ax3.plot(np.arange(len(X[0])), X[2, :], color='r', label='Velocity')
        self.ax3.plot(np.arange(len(X[0])), X[3, :], color='g', label='Yaw')
        self.ax3.set_ylabel('Values')
        self.ax3.set_xlabel('Time')
        self.ax3.set_title('Traj',fontsize=18)
        self.ax3.grid()
        self.ax3.legend()
        plt.pause(0.001)

