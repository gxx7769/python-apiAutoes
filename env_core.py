import numpy as np


class EnvCore(object): # 这段代码定义了一个名为EnvCore的类，表示一个智能体环境
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 14  # 设置智能体的观测纬度
        self.action_dim = 5  # 设置智能体的动作纬度，这里假定为一个五个纬度的

    def reset(self): # reset()方法用于重置环境并返回一个初始观察。
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        sub_agent_obs = [] # 首先创建一个空列表sub_agent_obs，
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, )) # 循环两次，每次生成一个形状为(14,)的随机观察，并将其添加到sub_agent_obs列表中
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs  # 返回sub_agent_obs列表作为初始观察。

    def step(self, actions): # step()方法用于执行一步动作并返回新的观察、奖励、完成状态和信息。这个方法接收一个动作列表作为输入，其中每个元素是一个形状为(5,)的随机动作。
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,))) # 为每个智能体生成一个形状为(14,)的随机观察，并将其添加到sub_agent_reward列表中
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({}) # 对于完成状态和信息，分别创建一个布尔值列表和一个空字典列表，并将它们添加到sub_agent_done和sub_agent_info列表中。

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
