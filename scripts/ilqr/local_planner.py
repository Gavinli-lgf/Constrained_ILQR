import numpy as np
import warnings
import pdb

"""
根据自车当前位置，从自车参考轨迹上离自车最近点开始取20个点，作为局部参考路径点，并使用5次多项式进行拟合
"""
class LocalPlanner:
    """
    Class which creates a desired trajectory based on the global plan
    Created by iLQR class based on the plan provided by the simulator
    这个类基于全局路径规划创建期望的轨迹。由 iLQR 类根据仿真器提供的参考轨迹创建。
    """

    """
    args : arguments.py中定义的参数; 
    """
    def __init__(self, args):
        self.args = args
        self.global_plan = None
        self.ego_state = None
        # self.control = control
    
    # ego_state:自车当前状态 (已转化为(5,3)大小的array)
    def set_ego_state(self, ego_state):
        self.ego_state = ego_state

    # global_plan:自车参考轨迹的位置(x,desired_y)数组，大小((map_lengthx+20) * 2)
    def set_global_planner(self, global_plan):
        """
        Sets the global plan of the ego vehicle
        """
        self.global_plan = np.asarray(global_plan) # array大小(70, 2)

    """
    输入: node:位置(x,y)
    输出: 通过欧氏距离计算自车参考路径self.global_plan上最接近node的点的索引
    """
    def closest_node(self, node):
        closest_ind = np.sum((self.global_plan - node)**2, axis=1)
        return np.argmin(closest_ind)

    """
    获取自车的局部参考路径：
    --在自车参考路径 self.global_plan 上取从自车当前位置最近的点及其后的self.args.number_of_local_wpts(默认20)个点作为局部参考路径点
    """
    def get_local_wpts(self):
        """
        Creates a local plan based on the waypoints on the global planner 
        """
        assert self.ego_state is not None, "Ego state was not set in the LocalPlanner"

        # Find index of waypoint closest to current pose
        closest_ind = self.closest_node([self.ego_state[0,0],self.ego_state[0,1]]) 
        # local_wpts = [[global_wpts[i,0],global_wpts[i,1]] for i in range(closest_ind, closest_ind + self.args.number_of_local_wpts)]
        return self.global_plan[closest_ind:closest_ind+self.args.number_of_local_wpts]

    """
    对自车局部路径参考点(默认20个)使用5次多项式进行拟合, 返回拟合后的局部路径参考点(2, 20)和拟合多项式的系数 coeffs
    (注: 拟合和样条曲线不同。拟合后的(x, new_y)满足5次多项式)
    """
    def get_local_plan(self):
        local_wpts = self.get_local_wpts()
        x = local_wpts[:,0]
        y = local_wpts[:,1]
        coeffs = np.polyfit(x, y, self.args.poly_order)
        new_y = np.polyval(np.poly1d(coeffs), x)

        # print(np.sum(np.abs(y - new_y)))
        # if (np.sum(np.abs(y - new_y)) > 0.5):
        #     pdb.set_trace()

        warnings.simplefilter('ignore', np.RankWarning)
        
        return np.vstack((x, new_y)).T, coeffs

    def get_orientation(self):
        """
        Gets the orientation of the path at the closest point on the local plan to be used by iLQR
        """
        raise NotImplementedError
    