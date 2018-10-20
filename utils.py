# coding=utf-8
import json
import csv

csv_file='/home/dilligencer/competition/baidu_dianshi/百度点石/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.csv'
txt_file='/home/dilligencer/competition/baidu_dianshi/百度点石/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.txt'

dict={'DESERT': 0, 'LAKE': 1, 'OCEAN':2, 'MOUNTAIN':3, 'FARMLAND':4, 'CITY':5}

source_file='/competition/guangdong/preprocess_12.json'
train_file='/competition/guangdong/train_2.txt'
val_file='/competition/guangdong/val_2.txt'
num=300
Flag=True

def generate_data_txt(source_file,train_file,val_file,Flag=True,num=300):
    with open(source_file, 'r') as fp:
        json_dict = json.loads(fp.read())

    with open(train_file,'w') as f1:
        with open(val_file,'w') as f2:
            for label ,data_list in json_dict.items():
                if int(label) != 11:
                    train_list = data_list[0:int(0.8 * len(data_list))]
                    val_list = data_list[int(0.8 * len(data_list)):]

                    if len(train_list) > num:
                        train_list = train_list[0:int(num * 1.2)]

                    elif len(train_list) <= num:
                        train_list = train_list * int(num / len(train_list))

                    for train_data in train_list:
                        row = train_data + ' ' + str(label) + '\n'
                        f1.write(row)
                    if val_list:
                        for val_data in val_list:
                            row = val_data + ' ' + str(label) + '\n'
                            f2.write(row)
                elif int(label) == 11:
                    train_list = data_list
                    for train_data in train_list:
                        row = train_data + ' ' + str(label) + '\n'
                        f1.write(row)

def csv2txt(csv_file,txt_file):
    with open(txt_file,'a') as f1:
        with open(csv_file,'r') as csvfile:
            reader=csv.reader(csvfile)
            for row in reader:
                txt='/home/dilligencer/competition/baidu_dianshi/百度点石/宽波段数据集-预赛训练集2000/预赛训练集-2000/'+row[0]+' '+ str(dict[row[1]]) +'\n'
                f1.write(txt)






def main():
    # generate_data_txt(source_file,train_file,val_file,Flag=Flag,num=num)
    csv2txt(csv_file=csv_file,txt_file=txt_file)



if __name__ == '__main__':
    main()