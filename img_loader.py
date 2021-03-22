from torch.utils.data import Dataset
import torch
import random
from PIL import Image, ImageChops
import numpy as np

def ImgOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    c.paste((0),(0,0,xoff,height))
    c.paste((0),(0,0,width,yoff))
    return c
def read_img(img_path):
    """
    keep reading until succeed
    :param img_path:
    :return:
    """
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset, sample='evenly', transform=None):
        self.dataset = dataset
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, bag = self.dataset[index]
        # print(img.shape,pid)
        # img = Image.fromarray(img)
        img = read_img(img_path)
        img = img.resize((64, 64))
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, bag
        # if self.sample == 'random':
        #     """
        #     read Image for train dataset
        #     """
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag
        #
        # elif self.sample == 'dense':
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag

        # else:
        #     raise KeyError("Unknown sample method: {}".format(self.sample))


class ImageDataset_aug(Dataset):

    def __init__(self, dataset, sample='evenly', transform=None):
        self.dataset = dataset
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, bag,rx,ry = self.dataset[index]
        # print(img.shape,pid)
        # img = Image.fromarray(img)
        img = read_img(img_path)
        img = img.resize((64, 64))
        img=ImgOfffSet(img,rx,ry)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, bag
        # if self.sample == 'random':
        #     """
        #     read Image for train dataset
        #     """
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag
        #
        # elif self.sample == 'dense':
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag

        # else:
        #     raise KeyError("Unknown sample method: {}".format(self.sample))




class ImageDataset_att(Dataset):

    def __init__(self, dataset, sample='evenly', transform=None):
        self.dataset = dataset
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, bag = self.dataset[index]
        # print(img.shape,pid)
        # img = Image.fromarray(img)
        img = read_img(img_path)
        img = img.resize((128,128))
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, bag
        # if self.sample == 'random':
        #     """
        #     read Image for train dataset
        #     """
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag
        #
        # elif self.sample == 'dense':
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag

        # else:
        #     raise KeyError("Unknown sample method: {}".format(self.sample))




