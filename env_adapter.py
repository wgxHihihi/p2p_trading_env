from p2p_trading_env.energy_trading_env import energy_trading_env
from gym.spaces import Box, Discrete
from dataclasses import dataclass
import numpy as np
from p2p_trading_env.env_logger import env_logger


@dataclass
class agent(object):
    name: str


class my_env:
    def __init__(self, run_dir, has_pv=2, no_pv=3, log_interval=100):
        self.env = energy_trading_env(has_pv, no_pv)
        self.n_agents = self.env.agents_num
        self.agents = [agent('building_%d' % i) for i in range(self.n_agents)]
        self.observation_space = [Box(low=float('-inf'), high=float('inf'), shape=(8,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=float('-inf'), high=float('inf'), shape=(8 * self.env.agents_num,)) for
                                        _ in range(self.n_agents)]
        self.action_space = [Box(low=float('-1'), high=float('1'), shape=(3,)) for _ in range(self.n_agents)]
        print('obs_space:', self.observation_space)
        print('act_space:', self.action_space)
        self.obs_shape = [8 for _ in range(self.n_agents)]
        self.action_shape = [3 for _ in range(self.n_agents)]
        self.run_dir = run_dir

        self.episode = 0
        self.log_interval = log_interval
        self.logger = env_logger(self.run_dir, self.n_agents)
        self.date = str()
        self.state = None

    def __del__(self):
        print('env close!')

    def seed(self, seed):
        self.env.seed(seed)

    def state_refactor(self, state):
        # print(state)
        common_state = [state[key] for key in
                        ['clearing_price', 'outdoor_temperature', 'buying_price', 'selling_price', 'time']]
        building_state = state['building_state']
        state = np.concatenate((building_state, [common_state] * self.n_agents), axis=1)
        # print(state)
        return state

    def reset(self):
        self.episode += 1
        ob = self.env.reset()
        self.date = ''.join(self.env.date[self.env.day_index * 96].split(' ')[0].split('/'))
        ob = self.state_refactor(ob)
        self.state = ob
        return ob

    def step(self, actions):
        actions = np.stack(actions)
        # actions = np.clip(actions, -1, 1)
        # print(actions)
        next_state, reward, done, info = self.env.step(actions)
        next_state = self.state_refactor(next_state)
        done = [done] * self.n_agents

        total_power = 0
        for b in self.env.buildings:
            total_power += b.p_total
        power_penalty = -0.5 * max(abs(total_power) - 2.4 * self.n_agents, 0)
        info = {'origin_reward': reward,
                'power_penalty': [power_penalty]
                }
        reward = [r + power_penalty for r in reward]
        if self.episode % self.log_interval == 0 and self.episode != 0:
            log_state = True
        else:
            log_state = False
        self.logger.log(log_state, self.env, self.state, actions, reward, done, info, self.date, self.episode)
        self.state = next_state

        return next_state, reward, done, info

    def close(self):
        self.__del__()
