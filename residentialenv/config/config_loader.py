"""
yaml文件转为属性类
"""
import yaml
import os
import pandas as pd


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class config_loader:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def dictToObj(self, dictObj):
        if not isinstance(dictObj, dict):
            return dictObj
        d = Dict()
        for k, v in dictObj.items():
            d[k] = self.dictToObj(v)
        return d

    def load_yaml(self, path):
        with open(path, encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def load_config(self, path):
        mydict = self.load_yaml(path)
        if not isinstance(mydict, dict):
            print('yaml format error')
            return mydict
        args = []
        for k, v in mydict.items():
            args.append(self.dictToObj(v))
        return args

    def load_building_configs(self):
        path_buildings = self.project_dir + r'/residentialenv/config/buildings_config.yaml'
        path_data_hasPV = self.project_dir + r'/residentialenv/training_data/building_data/hasPV'
        path_data_noPV = self.project_dir + r'/residentialenv/training_data/building_data/noPV'
        data_hasPV = []
        data_noPV = []
        for file in os.listdir(path_data_hasPV):
            file_index = file.split('_')[0].split('e')[1]
            data_hasPV.append([file_index,path_data_hasPV + '/' + file])
        for file in os.listdir(path_data_noPV):
            file_index = file.split('_')[0].split('e')[1]
            data_noPV.append([file_index,path_data_noPV + '/' + file])

        buildingargs = self.load_config(path_buildings)
        return buildingargs, data_hasPV, data_noPV

    def load_outdoor_temperature(self):
        T_out_path = self.project_dir + r'/residentialenv/training_data/temp_30min.csv'
        tempdata = pd.read_csv(T_out_path)
        T_out = tempdata['T_out']
        date = tempdata['time']
        return T_out, date

    def load_tou_prices(self):
        tou_price_path = self.project_dir + r'/residentialenv/training_data/price.csv'
        toudata = pd.read_csv(tou_price_path)
        tou_price = toudata['time-of-use']
        return tou_price
# path = '../config/buildings_config.yaml'
# config = load_yaml(path)
# print(config)
# buildingargs = load_buildings_config(config)
# args = dictToObj(config)
# # print(buildingargs[0].del_Tac)
# print(len(buildingargs))
