#coding=utf-8

import json
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from augmentation import *
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')


class Dataset2D_1(Dataset):
    def __init__(self, path, phase='train', shape=[512, 512], ratio=0.8):
        super(Dataset2D_1, self).__init__()
        self.phase = phase
        self.shape = shape

        with open(path, 'r') as fp:
            json_dict=json.loads(fp.read())

        self.file_dict = {}
        if self.phase == 'train':
            for key, item in json_dict.items():
                key = int(key)
                length = int(len(item)*ratio)
                self.file_dict[key] = item[:length]

        elif self.phase == 'val':
            for key, item in json_dict.items():
                length = int(len(item)*ratio)
                self.file_dict[key] = item[length:]
        else:
            print ('invalid phase.')

    def __getitem__(self, item):
        imgs, targets = [], []


        for key in self.file_dict.keys():
            length = len(self.file_dict[key])
            file = self.file_dict[key][item % length]
            # img =Image.open(file)
            img=np.load(file)
            img = Image.fromarray(np.uint8(img))
            img = img.resize(size=[256,256])
            # original_img=img
            flag = np.random.randint(0, 2)
            if flag:
                img = DataAugmentation.randomFlip(img)
            img = DataAugmentation.randomRotation(img)
            # img = DataAugmentation.randomGaussian(img)
            img = DataAugmentation.randomColor(img)
            img = np.asarray(img)

            # plt.subplot(121)
            # plt.imshow(original_img)
            # plt.subplot(122)
            # plt.imshow(img)
            # plt.show()

            imgs.append(img)
            targets.append(key)
        imgs=np.asarray(imgs)
        targets=np.asarray(targets)
        return imgs,targets

    def __len__(self):
        return len(self.file_dict[0])   #数据以正常数据遍历一次为一个epoch


if __name__ == '__main__':
    # path = '/competition/guangdong/preprocess_pickle.json'
    path = '/competition/guangdong/preprocess.json'
    dataset = Dataset2D(path, phase='train', shape=[512, 512], ratio=0.8)
    data_loader = DataLoader(dataset, batch_size=1,num_workers=4, shuffle=False)
    for data, target in data_loader:
        print (data.shape)

        # data = data.view(-1, 3, data.shape[-3], data.shape[-2])
        # data = data.type(torch.FloatTensor)
        # data = Variable(data.cuda())
        #
        # target=torch.squeeze(target)
        # target = target.type(torch.LongTensor)
        # target = Variable(target.cuda())
