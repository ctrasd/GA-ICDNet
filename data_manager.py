import os
import glob
import sys
import numpy as np
import os.path as osp
import random
import csv
from PIL import Image

class OULP_bag_txt(object):

    def __init__(self,root,label_file_path,cooperative=True):
        self.root =root
        self.tag = np.zeros(100000)
        self.num_train_pids = 27059
        self.cooperative=cooperative
        #labels_path = osp.join('/home/mxy/mxy/workspace/imdb/OU_LP_Bag', 'id_list.csv')
        labels_path=label_file_path
        line = 0
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row in reader:
                line += 1
                if line < 4:
                    continue
                for i in range(5,11,1):
                    if row[i] == '1':
                        self.tag[line - 3] = i-4

        train, test, val = self.__split_data()

        self.train = self.__process_data(train)
        self.test_probe, self.test_gallery = self.__process_dense_data(test)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)

    def __split_data(self):
        train_num = 27059
        # test_num = 2000
        test_num = 27058
        ids_list=[]
        with open('train_id.txt','r') as f:
            for row in f:
                row=row.strip()
                ids_list.append(row)
        with open('test_id.txt','r') as f:
            for row in f:
                row=row.strip()
                ids_list.append(row)

        #ids_list = os.listdir(self.root)
        print(len(ids_list))
        print("before filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list)))
        ids_list_filter = []

        for idx in range(len(ids_list)):
            #print(idx)
            if ids_list[idx][0] == '.':
                continue
            id_path = osp.join(self.root, ids_list[idx])
            id_list = os.listdir(id_path)
            id_list.sort()
            if len(id_list) < 2 or id_list[0][len(id_list[0])-5] != '1':
                continue
            ids_list_filter.append(ids_list[idx])
        print("after filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list_filter)))

        #random.shuffle(ids_list_filter)
        #ids_list_filter=random.shuffle()
        return ids_list_filter[0:train_num], ids_list_filter[train_num: train_num + test_num], \
               ids_list_filter[train_num + test_num:]

    def __process_dense_data(self, dataset):
        probe = []
        gallery = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            choose = [0,np.random.permutation(len(img_names)-1)[0]+1]

            img_paths.sort()
            if self.cooperative==True:
                choose_probe = img_paths[0]
                choose_gallery = img_paths[1]
            else:
                flag=np.random.randint(2)
                choose_probe = img_paths[choose[flag]]
                choose_gallery = img_paths[choose[1-flag]]
            probe_bag = 0
            gallery_bag = 0
            if choose_probe[len(choose_probe)-5] == '1' and self.tag[int(id_name)] >= 1:
                probe_bag = int(self.tag[int(id_name)])
            if choose_gallery[len(choose_gallery)-5] == '1' and self.tag[int(id_name)] >= 1:
                gallery_bag = int(self.tag[int(id_name)])
            probe.append((choose_probe, int(id_name), probe_bag))
            gallery.append((choose_gallery, int(id_name), gallery_bag))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset):
        tracklets = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            img_paths.sort()
            # img_paths = img_paths[0:2]
            # print(img_paths)

            for img_path in img_paths:
                bag = 0
                if img_path[len(img_path)-5] == '1' and self.tag[int(id_name)] >= 1:
                    bag = int(self.tag[int(id_name)])
                    # bag = self.tag[int(id_name)]
                tracklets.append((img_path, idx, bag))
        print("loading train over")
        return tracklets


def init_dataset_txt(root,label_file_path,cooperative):
    return OULP_bag_txt(root,label_file_path,cooperative)



class OULP_bag(object):

    def __init__(self,root,label_file_path,cooperative=True):
        self.root = root
        self.tag = np.zeros(100000)
        self.num_train_pids = 29097
        self.cooperative=cooperative
        #labels_path = osp.join('/home/mxy/mxy/workspace/imdb/OU_LP_Bag', 'id_list.csv')
        labels_path=label_file_path
        line = 0
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row in reader:
                line += 1
                if line < 4:
                    continue
                for i in range(5,11,1):
                    if row[i] == '1':
                        self.tag[line - 3] = i-4

        train, test, val = self.__split_data()

        self.train = self.__process_data(train)
        self.test_probe, self.test_gallery = self.__process_dense_data(test)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)

    def __split_data(self):
        train_num = 29097
        #test_num = 6000
        test_num = 29102
        ids_list = os.listdir(self.root)
        print(len(ids_list))
        print("before filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list)))
        ids_list_filter = []

        for idx in range(len(ids_list)):
            #print(idx)
            if ids_list[idx][0] == '.':
                continue
            id_path = osp.join(self.root, ids_list[idx])
            id_list = os.listdir(id_path)
            id_list.sort()
            if len(id_list) < 2 or id_list[0][len(id_list[0])-5] != '1':
                continue
            ids_list_filter.append(ids_list[idx])
        print("after filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list_filter)))
        r=random.random
        random.seed(1998326)
        random.shuffle(ids_list_filter,random=r)
        #ids_list_filter=random.shuffle()
        return ids_list_filter[0:train_num], ids_list_filter[train_num: train_num + test_num], \
               ids_list_filter[train_num + test_num:]

    def __process_dense_data(self, dataset):
        probe = []
        gallery = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            choose = [0,np.random.permutation(len(img_names)-1)[0]+1]

            img_paths.sort()
            if self.cooperative==True:
                choose_probe = img_paths[0]
                choose_gallery = img_paths[1]
            else:
                flag=np.random.randint(2)
                choose_probe = img_paths[choose[flag]]
                choose_gallery = img_paths[choose[1-flag]]
            probe_bag = 0
            gallery_bag = 0
            if choose_probe[len(choose_probe)-5] == '1' and self.tag[int(id_name)] >= 1:
                probe_bag = int(self.tag[int(id_name)])
            if choose_gallery[len(choose_gallery)-5] == '1' and self.tag[int(id_name)] >= 1:
                gallery_bag = int(self.tag[int(id_name)])
            probe.append((choose_probe, int(id_name), probe_bag))
            gallery.append((choose_gallery, int(id_name), gallery_bag))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset):
        tracklets = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            img_paths.sort()
            # img_paths = img_paths[0:2]
            # print(img_paths)

            for img_path in img_paths:
                bag = 0
                if img_path[len(img_path)-5] == '1' and self.tag[int(id_name)] >= 1:
                    bag = int(self.tag[int(id_name)])
                    # bag = self.tag[int(id_name)]
                tracklets.append((img_path, idx, bag))
        print("loading train over")
        return tracklets


def init_dataset(root,label_file_path,cooperative):
    return OULP_bag(root,label_file_path,cooperative)


class OULP_bag_ver(object):

    def __init__(self,root,label_file_path,cooperative=True):
        self.root = root
        self.tag = np.zeros(100000)
        self.num_train_pids = 29097
        self.cooperative=cooperative
        #labels_path = osp.join('/home/mxy/mxy/workspace/imdb/OU_LP_Bag', 'id_list.csv')
        labels_path=label_file_path
        line = 0
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row in reader:
                line += 1
                if line < 4:
                    continue
                for i in range(5,11,1):
                    if row[i] == '1':
                        self.tag[line - 3] = i-4

        train, test, val = self.__split_data()

        self.train = self.__process_data(train)
        self.test_probe, self.test_gallery = self.__process_dense_data(test)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)

    def __split_data(self):
        train_num = 29097
        #test_num = 10000
        test_num = 29102
        ids_list = os.listdir(self.root)
        print(len(ids_list))
        print("before filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list)))
        ids_list_filter = []

        for idx in range(len(ids_list)):
            #print(idx)
            if ids_list[idx][0] == '.':
                continue
            id_path = osp.join(self.root, ids_list[idx])
            id_list = os.listdir(id_path)
            id_list.sort()
            if len(id_list) < 2 or id_list[0][len(id_list[0])-5] != '1':
                continue
            ids_list_filter.append(ids_list[idx])
        print("after filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list_filter)))
        r=random.random
        random.seed(326)
        random.shuffle(ids_list_filter,random=r)
        #ids_list_filter=random.shuffle()
        return ids_list_filter[0:train_num], ids_list_filter[train_num: train_num + test_num], \
               ids_list_filter[train_num + test_num:]

    def __process_dense_data(self, dataset):
        probe = []
        gallery = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            choose = [0,np.random.permutation(len(img_names)-1)[0]+1]

            img_paths.sort()
            if self.cooperative==True:
                choose_probe = img_paths[0]
                choose_gallery = img_paths[1]
            else:
                flag=np.random.randint(2)
                choose_probe = img_paths[choose[flag]]
                choose_gallery = img_paths[choose[1-flag]]
            probe_bag = 0
            gallery_bag = 0
            if choose_probe[len(choose_probe)-5] == '1' and self.tag[int(id_name)] >= 1:
                probe_bag = int(self.tag[int(id_name)])
            if choose_gallery[len(choose_gallery)-5] == '1' and self.tag[int(id_name)] >= 1:
                gallery_bag = int(self.tag[int(id_name)])
            probe.append((choose_probe, int(id_name), probe_bag))
            gallery.append((choose_gallery, int(id_name), gallery_bag))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset):
        tracklets = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            img_paths.sort()
            # img_paths = img_paths[0:2]
            # print(img_paths)

            for img_path in img_paths:
                bag = 0
                if img_path[len(img_path)-5] == '1' and self.tag[int(id_name)] >= 1:
                    bag = int(self.tag[int(id_name)])
                    # bag = self.tag[int(id_name)]
                tracklets.append((img_path, idx, bag))
        print("loading train over")
        return tracklets


def init_dataset_ver(root,label_file_path,cooperative):
    return OULP_bag_ver(root,label_file_path,cooperative)




class OULP_bag_aug(object):

    def __init__(self,root,label_file_path,cooperative=True):
        self.root = root
        self.tag = np.zeros(100000)
        self.num_train_pids = 29097
        self.cooperative=cooperative
        #labels_path = osp.join('/home/mxy/mxy/workspace/imdb/OU_LP_Bag', 'id_list.csv')
        labels_path=label_file_path
        line = 0
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row in reader:
                line += 1
                if line < 4:
                    continue
                for i in range(5,11,1):
                    if row[i] == '1':
                        self.tag[line - 3] = i-4

        train, test, val = self.__split_data()

        self.train = self.__process_data(train)
        self.test_probe, self.test_gallery = self.__process_dense_data(test)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)

    def __split_data(self):
        train_num = 29097
        # test_num = 2000
        test_num = 29102
        ids_list = os.listdir(self.root)
        print(len(ids_list))
        print("before filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list)))
        ids_list_filter = []

        for idx in range(len(ids_list)):
            #print(idx)
            if ids_list[idx][0] == '.':
                continue
            id_path = osp.join(self.root, ids_list[idx])
            id_list = os.listdir(id_path)
            id_list.sort()
            if len(id_list) < 2 or id_list[0][len(id_list[0])-5] != '1':
                continue
            ids_list_filter.append(ids_list[idx])
        print("after filter the len(id's imgs) < 2, the number of id is %d"%(len(ids_list_filter)))

        random.shuffle(ids_list_filter)
        #ids_list_filter=random.shuffle()
        return ids_list_filter[0:train_num], ids_list_filter[train_num: train_num + test_num], \
               ids_list_filter[train_num + test_num:]

    def __process_dense_data(self, dataset):
        probe = []
        gallery = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            choose = [0,np.random.permutation(len(img_names)-1)[0]+1]

            img_paths.sort()
            if self.cooperative==True:
                choose_probe = img_paths[0]
                choose_gallery = img_paths[1]
            else:
                flag=np.random.randint(2)
                choose_probe = img_paths[choose[flag]]
                choose_gallery = img_paths[choose[1-flag]]
            probe_bag = 0
            gallery_bag = 0
            if choose_probe[len(choose_probe)-5] == '1' and self.tag[int(id_name)] >= 1:
                probe_bag = int(self.tag[int(id_name)])
            if choose_gallery[len(choose_gallery)-5] == '1' and self.tag[int(id_name)] >= 1:
                gallery_bag = int(self.tag[int(id_name)])
            probe.append((choose_probe, int(id_name), probe_bag))
            gallery.append((choose_gallery, int(id_name), gallery_bag))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset):
        tracklets = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            img_paths.sort()
            # img_paths = img_paths[0:2]
            # print(img_paths)
            rx=random.randint(-5,5)
            ry=random.randint(-5,5)
            for img_path in img_paths:
                bag = 0
                if img_path[len(img_path)-5] == '1' and self.tag[int(id_name)] >= 1:
                    bag = int(self.tag[int(id_name)])
                    # bag = self.tag[int(id_name)]
                tracklets.append((img_path, idx, bag,rx,ry))
        print("loading train over")
        return tracklets


def init_dataset_aug(root,label_file_path,cooperative):
    return OULP_bag_aug(root,label_file_path,cooperative)



class OULP_bag_beta(object):

    def __init__(self,root,cooperative=True):
        self.root = root
        self.tag = np.zeros(100000)
        self.num_train_pids = 29097
        self.cooperative=cooperative
        #labels_path = osp.join('/home/mxy/mxy/workspace/imdb/OU_LP_Bag', 'id_list.csv')
        
        train_probe, train_gallery,test_probe,test_gallery = self.__split_data()

        self.train = self.__process_data(train_probe,train_gallery)
        self.test_probe, self.test_gallery = self.__process_dense_data(test_probe,test_gallery)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)

    def __split_data(self):
        train_num = 29097
        #test_num = 6000
        test_num = 29102
        ids_train_gallery=[]
        ids_train_probe=[]
        ids_test_gallery=[]
        ids_test_probe=[]
        with open('./GaitLPbag/gallery_train_list.txt','r') as f:
            for row in f:
                row=row.strip().split(' ')
                ids_train_gallery.append([row[0],row[1]])
        with open('./GaitLPbag/probe_train_list.txt','r') as f:
            for row in f:
                row=row.strip().split(' ')
                ids_train_probe.append([row[0],row[1]])

        with open('./GaitLPbag/gallery_test_list.txt','r') as f:
            for row in f:
                row=row.strip().split(' ')
                ids_test_gallery.append([row[0],row[1]])

        with open('./GaitLPbag/probe_test_list.txt','r') as f:
            for row in f:
                row=row.strip().split(' ')
                ids_test_probe.append([row[0],row[1]])



        return ids_train_probe,ids_train_gallery,ids_test_probe,ids_test_gallery

    def __process_dense_data(self, dataset_probe,dataset_gallery):
        probe = []
        gallery = []
        for idx in range(len(dataset_probe)):
            id_name_probe,id_path_probe = dataset_probe[idx]
            id_name_gallery,id_path_gallery = dataset_gallery[idx]
            print(id_name_probe,id_name_gallery)
            probe.append((self.root+id_path_probe, int(id_name_probe), 1))
            gallery.append((self.root+id_path_gallery, int(id_name_gallery), 0))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset_probe,dataset_gallery):
        tracklets = []
        #length=len(dataset_probe)
        for idx in range(len(dataset_probe)):
            id_name_probe,id_path_probe = dataset_probe[idx]
            id_name_gallery,id_path_gallery = dataset_gallery[idx]
            print(id_name_probe,id_name_gallery)
            tracklets.append((self.root+id_path_probe, int(id_name_probe), 1))
            tracklets.append((self.root+id_path_gallery, int(id_name_gallery), 0))

        print("loading train over")
        return tracklets


def init_dataset_beta(root,cooperative):
    return OULP_bag_beta(root,cooperative)


