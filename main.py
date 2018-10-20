#coding=utf-8
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from dataset2 import Dataset2D_2
from Inception import *
from densenet import DenseNet_1
from densenet_2 import densenet121
import csv
from SE_RseNet import *


# test_path='/competition/guangdong/guangdong_round1_test_a_20180916/'
train_path='/home/dilligencer/competition/baidu_dianshi/百度点石/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.txt'
val_path='/home/dilligencer/competition/baidu_dianshi/百度点石/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.txt'
model_dir='/home/dilligencer/competition/baidu_dianshi/百度点石/save_model/'


load_model=False
BATCH_SIZE=32
val_batch=1
lr=0.0001
Epoch=100
num_workers=4
shape=[256,256]
NC=6
val_NC=6
LOSS_THRESH=0.0005

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# densenet=DenseNet_1(growthRate=12, depth=50, reduction=0.5,
#                             bottleneck=True, nClasses=14)
#
# densenet=densenet121(pretrained=False)
# # print(Inception_V4.linear.in_features)
# densenet.classifier=torch.nn.Linear(65536,NC)
se_resnet101=se_resnet152(num_classes=NC)
# print(se_resnet101.fc.in_features)
se_resnet101.avgpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
se_resnet101.fc=torch.nn.Linear(2048*4*4,NC)


def main(train_loader,lr=0.001,num_e=0):
    torch.cuda.manual_seed(1)
    se_resnet101.cuda()
    if load_model:
        checkpoint= torch.load(model_dir + '{}.ckpt'.format(num_e))
        se_resnet101.load_state_dict(checkpoint['state_dict'])
        print ('Loading model~~~~~~~~~~', num_e)

    for epoch in range(Epoch):
        if (epoch+1) % 20 ==0:
            lr = lr*1.0 /10.0
        se_resnet101.train()
        optimizer = torch.optim.Adam(params=se_resnet101.parameters(),lr=lr,weight_decay=1e-5)
        for i,(data,target) in enumerate(train_loader):
            data = data.view(-1, 3, data.shape[-2], data.shape[-1])
            data = data.type(torch.FloatTensor)
            data = Variable(data.cuda())

            target = torch.squeeze(target)
            target = target.type(torch.LongTensor)
            target = Variable(target.cuda())
            # weight=torch.tensor([1.5,1.0,5.0,1.0,1.0,1.5,1.4,1.0,1.0,1.0,5.0,5.0])
            # weight=weight.cuda()

            optimizer.zero_grad()
            output=se_resnet101(data)
            # output=nn.Sigmoid()(output)
            # print ('loss>>>>',output)
            # print ('target>>>>',target)
            #loss=nn.CrossEntropyLoss(weight=weight)(output,target)
            loss = nn.CrossEntropyLoss()(output, target)
            if loss.cpu().data.numpy() > LOSS_THRESH:
                loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print (loss.cpu().data.numpy())

        if (epoch+1) % 5 == 0:
            state_dict=se_resnet101.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save(
                {
                    'epoch': epoch,
                    'save_dir': model_dir,
                    'state_dict': state_dict,
                }, os.path.join(model_dir, '%d.ckpt' % epoch))

            val(val_train_loader,epoch=epoch,val=True)
            # val(val_loader, epoch=epoch, val=True)


def val(val_loader,epoch=0,val=False):
    if  val:
        se_resnet101.cuda()
        checkpoint = torch.load(model_dir + '%d.ckpt' % epoch)
        se_resnet101.load_state_dict(checkpoint['state_dict'])

    se_resnet101.eval()
    total = np.zeros(val_NC).astype(float)
    correct = np.zeros(val_NC).astype(float)
    pre = np.zeros(val_NC).astype(float)


    for i,(data,target) in enumerate(val_loader):
        data = data.view(-1, 3, data.shape[-2], data.shape[-1])
        data = data.type(torch.FloatTensor)
        data = Variable(data.cuda())

        # target = target.type(torch.LongTensor)
        # target = Variable(target.cuda())
        # target = torch.squeeze(target)
        target = target.numpy()[0]

        # print target

        output=se_resnet101(data).cpu()
        # print output.shape
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.numpy()
        # print predicted
        #predicted[predicted>11] =11
        #target[target>11] =11

        # print (target.shape,predicted.shape)
        for i in range(len(predicted)):
            if predicted[i] == target[i]:
                correct[target[i]] += 1
            # elif predicted[i] != target[i]:
            #     print('predict>>>',predicted[i] , 'target>>>>',target[i])
            total[target[i]] += 1
            pre[predicted[i]] += 1

    precision = (correct / total).sum() / val_NC
    print('val epoch>>>>>>>',epoch)
    print ('precision = ', precision)
    print ('prediction ', pre)
    print ('correct    ', correct)
    print ('total      ', total)


def test(test_loader,epoch,result_file):
    se_resnet101.cuda()
    checkpoint = torch.load(model_dir + '%d.ckpt' % epoch)
    se_resnet101.load_state_dict(checkpoint['state_dict'])
    se_resnet101.eval()

    for data,name in test_loader:
        data = data.view(-1, 3, data.shape[-2], data.shape[-1])
        data = data.type(torch.FloatTensor)
        data = Variable(data.cuda())
        output = se_resnet101(data).cpu()
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.numpy()
        # predicted[predicted>11] = 11

        with open(result_file,'a') as fp:
            csv_write = csv.writer(fp)
            for i in range(len(name)):
                if predicted[i] > 0:
                    prediction = 'defect' + str(predicted[i])
                else:
                    prediction = 'norm'
                tmp = [name[i], prediction]
                csv_write.writerow(tmp)


if __name__ == '__main__':
    train_dataset=Dataset2D_2(train_path,phase='train',shape=shape)
    train_loader=DataLoader(train_dataset,batch_size = BATCH_SIZE,num_workers=num_workers,shuffle=True)

    val_train_loader=DataLoader(train_dataset, batch_size=val_batch, num_workers=num_workers, shuffle=True)

    val_dataset = Dataset2D_2(val_path, phase='val', shape=shape)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, num_workers=num_workers, shuffle=True)
    main(train_loader,lr=lr,num_e=0)
    # for e in [23]:
    #     val(val_loader,epoch=e,val= True)
    # test_dataset=Dataset2D_2(test_path,phase='test',shape=shape)
    # test_loader=DataLoader(test_dataset,batch_size=val_batch,num_workers=num_workers,shuffle=False)
    # for e in [26]:
    #     result_file = ('/competition/guangdong/result_new_'+'%d.csv' % e)
    #     if not os.path.exists(result_file):
    #         os.mknod(result_file)
    #     test(test_loader,epoch=e,result_file=result_file)
