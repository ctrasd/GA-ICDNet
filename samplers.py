from __future__ import absolute_import
import torch

import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    randomly sample N identities
    randomly sample k = 2 instances(image) for one identities
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.bag_dic = defaultdict(list)

        for index, (_, pid , bag) in enumerate(data_source):
            self.index_dic[pid].append(index)
            self.bag_dic[pid].append(bag)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            bag = self.bag_dic[pid]
            is_index = []
            find = False
            for j in range(len(bag)):
                if bag[j] == 0 and find == False:
                    ret.extend([t[j]])
                    find = True
                else:
                    is_index.append(t[j])
            # t = np.random.choice(t, size=self.num_instances, replace=False)
            t1 = np.random.choice(is_index, size=1, replace=False)
            ret.extend(t1)
        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_identities

class RandomIdentitySampler_aug(Sampler):
    """
    randomly sample N identities
    randomly sample k = 2 instances(image) for one identities
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.bag_dic = defaultdict(list)
        for index, (_, pid , bag,rx,ry) in enumerate(data_source):
            self.index_dic[pid].append(index)
            self.bag_dic[pid].append(bag)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            bag = self.bag_dic[pid]
            is_index = []
            find = False
            for j in range(len(bag)):
                if bag[j] == 0 and find == False:
                    ret.extend([t[j]])
                    find = True
                else:
                    is_index.append(t[j])
            # t = np.random.choice(t, size=self.num_instances, replace=False)
            t1 = np.random.choice(is_index, size=1, replace=False)
            ret.extend(t1)
        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_identities

