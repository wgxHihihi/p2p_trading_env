import numpy as np
import random
import pandas as pd

timeslots = 48


class building_env:
    """
    用户环境：ESS，随机负荷
    状态：ev_soc，固定负荷，总负荷
    动作：储能功率
    奖励：
    """

    def __init__(self, data_path, building_args, home_id, home_type, start_index):
        self.delta_t = 24 / timeslots
        self.index = 0
        self.home_id = home_id
        self.home_type = home_type

        # contract of half-hourly ahead energy trading
        self.contract = list()

        # ESS
        self.soc = building_args.soc_ini
        self.ess_p_max = building_args.ess_p_max
        self.soc_max = building_args.soc_max
        self.soc_min = building_args.soc_min
        self.ess_e_max = building_args.ess_e_max
        self.a_ess = 0

        # Nonshiftable Apps
        self.fixed_Power, self.time_array = self.data_loader(data_path)  # 随机的固定负荷
        self.time = str()
        # 总功率
        self.p_total = 0

        # 状态
        self.state = np.array([self.soc, self.fixed_Power[0], 0, 0, 0])

        self.action_pre = np.zeros(1)

    def seed(self, seed):
        random.seed(seed)

    def reset(self, start_index):
        self.contract = list()
        self.p_total = self.fixed_Power[start_index]
        self.time = self.time_array[start_index]

        self.state = np.hstack(
            [self.soc, np.array([self.fixed_Power[start_index], self.p_total])])

        return self.state

    @staticmethod
    def data_loader(path):
        fixed_power_data = pd.read_csv(path)
        return np.array(fixed_power_data['grid']), np.array(fixed_power_data['local_15min'])

    def soc_constraint(self, action):
        # 电池空了之后只允许充电
        action = np.clip(action, -1, 1)
        power = action * self.ess_p_max
        if power >= 0:
            p_new = np.clip(power, 0,
                            min(self.ess_p_max, ((self.soc_max - self.soc) * self.ess_e_max / 0.95) / self.delta_t))
        else:
            p_new = np.clip(power,
                            max(-self.ess_p_max, ((self.soc_min - self.soc) * self.ess_e_max * 0.95) / self.delta_t), 0)
        return p_new

    def __SoC(self, soc_t, power):
        if power >= 0:
            soc_t_1 = soc_t + 0.95 * power * self.delta_t / self.ess_e_max
        else:
            soc_t_1 = soc_t + (1 / 0.95) * power * self.delta_t / self.ess_e_max
        soc_t_1 = np.clip(soc_t_1, 0, self.soc_max)
        return soc_t_1

    def step(self, action, start_index, time):
        """
        :param action:AC,ESS
        :param start_index: start index
        :param time: time slot
        :return: next state
        """
        self.a_ess = action
        action_new = self.soc_constraint(action)
        self.action_pre = action_new

        self.p_total = 0
        self.p_total += action_new + self.fixed_Power[start_index + time]
        self.time = self.time_array[start_index + time]

        self.soc = self.__SoC(self.soc, action_new)

        """
        状态：储能电量，固定负荷，总负荷
        """
        self.state = np.hstack([self.soc, np.array([self.fixed_Power[start_index + time], self.p_total])])

        return self.state

    def clearing_fn(self, buy_price, sold_price):
        """
        :param buy_price:
        :param sold_price:
        :return:
        """
        # electricity cost
        if self.p_total >= 0:
            price = buy_price
        else:
            price = sold_price
        e_cost = self.p_total * self.delta_t * price
        r = - e_cost

        return r, e_cost
