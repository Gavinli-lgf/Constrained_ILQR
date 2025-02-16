from matplotlib.pylab import f
import numpy as np
import pdb

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)

class Model:
    """
    A vehicle model with 4 dof. 
    State - [x, y, vel, theta]
    Control - [acc, yaw_rate]
    """
    def __init__(self, args):
        self.wheelbase = args.wheelbase
        self.steer_min = args.steer_angle_limits[0]
        self.steer_max = args.steer_angle_limits[1]
        self.accel_min = args.acc_limits[0]
        self.accel_max = args.acc_limits[1]
        self.max_speed = args.max_speed
        self.Ts = args.timestep
        self.N = args.horizon
        self.z = np.zeros((self.N))
        self.o = np.ones((self.N))
        
    # 状态转移方程：使用车辆运动学方程，根据当前状态 state 和控制输入 control ，计算下一个状态 next_state
    def forward_simulate(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values(对输入进行限幅)
        control[0] = np.clip(control[0], self.accel_min, self.accel_max)
        control[1] = np.clip(control[1], state[2]*tan(self.steer_min)/self.wheelbase, state[2]*tan(self.steer_max)/self.wheelbase)
        
        # 使用车辆运动学方程，由当前状态和控制输入计算下一个状态(应该先对v限幅，再计算x,y)
        next_state = np.array([state[0] + cos(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
                               state[1] + sin(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
                               np.clip(state[2] + control[0]*self.Ts, 0.0, self.max_speed),
                               state[3] + control[1]*self.Ts])  # wrap angles between 0 and 2*pi - Gave me error
        return next_state

    """
    功能：求状态转移方程一阶泰勒展开后X_k+1 = AX_k + BU_k中的A，即df/dx
    输入: noimnal trajectory信息"速度,朝向,加速度"(不包含初始状态):velocity_val(1,40), theta(1,40), acceleration_val(1,40);
    输出: A(4,4,40): 状态转移矩阵A
    注: 公式参照论文中的式(17)和(18)
    """
    def get_A_matrix(self, velocity_vals, theta, acceleration_vals):
        """
        Returns the linearized 'A' matrix of the ego vehicle 
        model for all states in backward pass. 
        """
        v = velocity_vals
        v_dot = acceleration_vals
        # A表面上看是一个4*4的矩阵,但是矩阵中的每个元素,实际上都是一个1*40子矩阵
        A = np.array([[self.o, self.z, cos(theta)*self.Ts, -(v*self.Ts + (v_dot*self.Ts**2)/2)*sin(theta)],
                      [self.z, self.o, sin(theta)*self.Ts,  (v*self.Ts + (v_dot*self.Ts**2)/2)*cos(theta)],
                      [self.z, self.z,             self.o,                                         self.z],
                      [self.z, self.z,             self.z,                                         self.o]])
        return A

    """
    功能：求状态转移方程一阶泰勒展开后X_k+1 = AX_k + BU_k中的B，即df/du
    输入: noimnal trajectory信息"朝向"(不包含初始状态):theta(1,40)
    输出: B(4,2,40): 控制矩阵B
    注: 公式参照论文中的式(17)和(18)
    """
    def get_B_matrix(self, theta):
        """
        Returns the linearized 'B' matrix of the ego vehicle 
        model for all states in backward pass. 
        """
        # B表面上看是一个4*2的矩阵,但是矩阵中的每个元素,实际上都是一个1*40子矩阵
        B = np.array([[self.Ts**2*cos(theta)/2,         self.z],
                      [self.Ts**2*sin(theta)/2,         self.z],
                      [         self.Ts*self.o,         self.z],
                      [                 self.z, self.Ts*self.o]])
        return B
