import pandas as pd
import numpy as np
import os


class env_logger:
    def __init__(self, save_path, agents_num):
        self.save_path = save_path / 'env_logs'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.record = []
        self.reward = []
        self.agents_num = agents_num
        self.reward_dis = np.zeros(agents_num + 2)

    def log(self, log_state, env, obs, act, rew, done, info, date, ep):
        if log_state:
            soc = obs[:, 0]
            common_state = obs[0, -5:]
            actions_pre = np.hstack(act)

            p_buildings = obs[:, 2]
            p_fixed = obs[:, 1]
            p_net = []
            for i in range(self.agents_num):
                p_net.append(env.buildings[i].p_total)
            p_net_total = [sum(p_net)]
            match_books = []
            for match in env.match_books:
                offerid = match.Offer.CreatorID
                bidid = match.Bid.CreatorID
                trading_price = match.trading_price
                quantity = match.Offer.Quantity
                match_log = [offerid, bidid, trading_price, quantity]
                match_books.append(match_log)
            time = [env.time]
            self.record.append((time + p_net_total + list(common_state) +
                                list(soc) + list(p_buildings) + list(p_fixed) +
                                p_net + list(actions_pre) + [match_books]))
        self.reward_dis += rew[:1] + info['origin_reward'] + info['power_penalty']
        if any(done):
            self.reward.append(self.reward_dis)
            if log_state:
                self.save_to_csv(date, ep)

            self.reward_dis = np.zeros(self.agents_num + 2)
            self.record = []

    def save_to_csv(self, date, ep):
        record_columns = ['time',
                          'p_net_total',
                          'clearing_price',
                          'outdoor_temperature',
                          'buying_price',
                          'selling_price',
                          'time_index']
        keys = ['soc', 'power', 'power_fix', 'p_net']
        for k in keys:
            record_columns += [k + '_%d' % i for i in range(self.agents_num)]
        for i in range(self.agents_num):
            record_columns += ['auc_price_%d' % i, 'auc_quantity_%d' % i, 'a_ess_%d' % i]
        record_columns += ['match_books']
        pd.DataFrame(self.record,
                     columns=record_columns
                     ).to_csv(self.save_path / 'record_{}_{}.csv'.format(ep, date))

        pd.DataFrame(self.reward,
                     columns=['total'] +
                             ['home%d' % i for i in range(self.agents_num)] +
                             ['power_penalty']
                     ).to_csv(self.save_path / 'r_{}.csv'.format(ep))
