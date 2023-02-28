import numpy as np
from double_auction import Market, Order
from residentialenv.single_home_env import building_env
from residentialenv.config.config_loader import *


class energy_trading_env:
    def __init__(self, has_pv=2, no_pv=3):
        self.Seed: int
        self.clock = 0
        self.state = dict()
        self.delta_t = 24 / 48
        self.time = str()

        self.config_loader = config_loader()
        # load outdoor temperature
        self.T_out, self.date = self.config_loader.load_outdoor_temperature()

        # init double auction market
        self.trading_market = Market()

        # init buildings
        ## real world data configs: need to modify
        self.day_index = 0
        self.time_bias = 0

        self.hasPV = has_pv
        self.noPV = no_pv
        print('------hasPV: %d, noPV: %d------' % (self.hasPV, self.noPV))
        self.buildings_args, self.building_hasPV, self.building_noPV = self.config_loader.load_building_configs()
        assert (self.hasPV <= len(self.building_hasPV))
        assert (self.noPV <= len(self.building_noPV))

        self.buildings = []
        for i in range(self.hasPV):
            k, v = self.building_hasPV[i]
            self.buildings.append(
                building_env(v, self.buildings_args[0], k, 'hasPV', self.day_index * 96 + self.time_bias))
        for i in range(self.noPV):
            k, v = self.building_noPV[i]
            self.buildings.append(
                building_env(v, self.buildings_args[0], k, 'noPV', self.day_index * 96 + self.time_bias))

        self.agents_num = len(self.buildings)

        # load tou price
        self.buying_price = self.config_loader.load_tou_prices()
        self.selling_price = self.buying_price * 0.6

        # match books
        self.match_books = []

    def seed(self, seed):
        self.Seed = seed
        for b in self.buildings:
            b.seed(seed)

    def reset(self):
        """
        :return: initial state
        """
        self.trading_market.clear()
        self.clock = 0
        self.match_books = []
        start_index = self.day_index * 96 + self.time_bias
        self.time = self.date[start_index]

        initial_state = dict()
        building_state = []
        for index, building in enumerate(self.buildings):
            building_state.append(building.reset(start_index))
        """
        need to do
        """
        initial_state['building_state'] = np.stack(building_state)
        initial_state['clearing_price'] = 0
        initial_state['outdoor_temperature'] = self.T_out[start_index]
        initial_state['buying_price'] = self.buying_price[start_index]
        initial_state['selling_price'] = self.selling_price[start_index]
        initial_state['time'] = 0

        self.state = initial_state

        return initial_state

    def step(self, actions):
        """
        :param actions: nparray: [[biding price, biding quality, storage action],[],[],...,[]]
        :return: next state, reward, done, info
        """
        start_index = self.day_index * 96 + self.time_bias
        self.clock += 1

        self.time = self.date[start_index + self.clock]

        auction_actions = actions[:, 0:2]
        storage_actions = actions[:, 2]

        # trade energy, sign contract
        # current tou price
        buy_price = self.state['buying_price']
        sell_price = self.state['selling_price']
        for i in range(len(auction_actions)):
            biding_price, quantity = auction_actions[i]
            biding_price = np.clip(biding_price, -1, 1)
            biding_price = (biding_price + 1) / 2 * (buy_price - sell_price) + sell_price
            neworder = Order(i, quantity < 0, abs(quantity), biding_price)
            self.trading_market.AddOrder(neworder)
        self.trading_market.MatchOrders()
        self.match_books = self.trading_market.Matches
        # # send contract to buildings
        # for index, match in enumerate(self.trading_market.Matches):
        #     offerID = match.Offer.CreatorID
        #     bidID = match.Bid.CreatorID
        #     self.buildings[offerID].contract.append(match)
        #     self.buildings[bidID].contract.append(match)

        # schedule the storage of the users
        next_state = dict()
        building_state = []
        for index, building in enumerate(self.buildings):
            building_state.append(building.step(storage_actions[index], start_index, self.clock))

        reward = self.reward_fn(buy_price, sell_price)
        next_state['building_state'] = np.stack(building_state)

        # clearing price
        clearing_price = self.trading_market.ComputeClearingPrice()
        next_state['clearing_price'] = clearing_price
        next_state['outdoor_temperature'] = self.T_out[start_index + self.clock]
        next_state['buying_price'] = self.buying_price[start_index + self.clock]
        next_state['selling_price'] = self.selling_price[start_index + self.clock]
        next_state['time'] = self.clock

        done = self.clock >= 48
        info = dict()
        self.state = next_state
        return next_state, reward, done, info

    def reward_fn(self, buy_price, sell_price):
        # clearing contract & calculate the power balance cost
        rewards = [0] * len(self.buildings)

        # contract clearing
        for match in self.trading_market.Matches:
            offerID = match.Offer.CreatorID
            bidID = match.Bid.CreatorID
            quantity = match.Bid.Quantity
            trading_price = match.trading_price

            bid_quantity = min(max(self.buildings[bidID].p_total * self.delta_t, 0), quantity)  # 买方可交易量
            offer_quantity = min(max(-self.buildings[offerID].p_total * self.delta_t, 0), quantity)  # 卖方可交易量
            trading_quantity = min(bid_quantity, offer_quantity)  # 实际成交量
            trading_power = trading_quantity / self.delta_t  # 实际成交量，按功率计

            trading_cost = trading_price * trading_quantity  # 交易费用，由买家支付给卖家
            bid_penalty = (offer_quantity - trading_quantity) * trading_price * 0.05  # 向卖方提供5%的违约金
            offer_penalty = (bid_quantity - trading_quantity) * trading_price * 0.05  # 向买方提供5%的违约金

            rewards[offerID] += trading_cost + bid_penalty - offer_penalty
            rewards[bidID] += -trading_cost - bid_penalty + offer_penalty

            self.buildings[offerID].p_total += trading_power
            self.buildings[bidID].p_total -= trading_power

        # banlance power with the grid
        for index, building in enumerate(self.buildings):
            price = buy_price if building.p_total > 0 else sell_price
            grid_cost = building.p_total * self.delta_t * price
            # print(building.p_total, grid_cost)
            rewards[index] += -grid_cost
        return rewards

# env = energy_trading_env(12)
# print(env.reset())
# print(env.step(np.array([[0.12, 2, 3], [0.08, 3, 3], [0.07, 2, 3], [0.06, -4, 3], [0.10, -2, 3]])))
