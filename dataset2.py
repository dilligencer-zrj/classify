#coding=utf-8
from glob import  glob
import json
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from augmentation import *
# from matplotlib import pyplot as plt
# plt.switch_backend('TkAgg')

def normiliaze(image,mean,std):
    mean=np.array(mean).reshape(3,1,1)
    std=np.array(std).reshape(3,1,1)
    return (image.transpose((2,0,1))/255. - mean) /std


class Dataset2D_2(Dataset):
    def __init__(self, path, phase='train', shape=[512, 512]):
        super(Dataset2D_2, self).__init__()
        self.phase = phase
        self.shape = shape
        if self.phase == 'train' or self.phase == 'val':
            with open(path,'r') as fp:
                self.data_list=fp.readlines()
        elif self.phase == 'test':
            files=glob(path+'*')
            self.data_list=sorted(files, key=lambda file:int(file.split('/')[-1].replace('.jpg', '')))



    def __getitem__(self, item):


        if self.phase == 'train' or self.phase =='val':

            row =self.data_list[item]
            file=row.strip().split(' ')[0]
            label=[int(row.strip().split(' ')[-1])]
            img =Image.open(file)
            # original_img = img
            # img=np.load(file)
            # img = Image.fromarray(np.uint8(img))
            # img=DataAugmentation.CenterCrop(img)
            if self.phase == 'train':
                flag_1 = np.random.randint(0, 2)
                flag_2 = np.random.randint(0, 2)
                if flag_1:
                    img = DataAugmentation.randomFlip_1(img)
                if flag_2:
                    img = DataAugmentation.randomFlip_2(img)
                img = img.resize(size=self.shape)
                img = DataAugmentation.randomRotation(img)
                # img = DataAugmentation.randomGaussian(img)
                # img = DataAugmentation.randomColor(img)
                img=np.asarray(img)
                img = normiliaze(img,[0.485,0.456,0.406],[0.229,0.224,0.225])

                # plt.subplot(121)
                # plt.imshow(original_img)
                # plt.subplot(122)
                # plt.imshow(img)
                # plt.show()

                return np.asarray(img), np.asarray(label)
            else:
                img = img.resize(size=self.shape)
                img = np.asarray(img)
                img = normiliaze(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                return np.asarray(img), np.asarray(label)

        elif self.phase == 'test':
            file=self.data_list[item]
            img = Image.open(file)
            # img = DataAugmentation.CenterCrop(img)
            img = img.resize(size=self.shape)
            img = np.asarray(img)
            img = normiliaze(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            target = file.split('/')[-1]

            return np.asarray(img),target


    def __len__(self):
        return len(self.data_list)   #数据以正常数据遍历一次为一个epoch


if __name__ == '__main__':
    # path = '/competition/guangdong/preprocess_pickle.json'
    # path = '/home/dilligencer/比赛数据/广东工业智造大数据创新大赛 - 智能算法赛/train.txt'
    path = '/competition/guangdong/train.txt'
    dataset = Dataset2D_2(path, phase='train', shape=[512, 512])
    data_loader = DataLoader(dataset, batch_size=8,num_workers=4, shuffle=True)
    for data, target in data_loader:
        print (data.shape)
        print (target.shape)

        # data = data.view(-1, 3, data.shape[-3], data.shape[-2])
        # data = data.type(torch.FloatTensor)
        # data = Variable(data.cuda())
        #
        # target=torch.squeeze(target)
        # target = target.type(torch.LongTensor)
        # target = Variable(target.cuda())
