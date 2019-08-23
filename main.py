import os
import torch as t
from torch import nn
from torchnet import meter
from torch.utils.data import DataLoader
import models
from config import opt
from tqdm import tqdm
from data import DataReader

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def train(**kwargs):

    '''
    训练
    '''
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    vis = Visulizer(opt.env)

    # step1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: 数据
    train_data = DataReader(opt.train_data_root, train = True)
    val_data = DataReader(opt.train_data_root, train = False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle = True, num_workers = opt.num_workers,)
    val_dataloader = DataLoader(val_data, opt.batch_size,shuffle = False,num_workers = opt.num_workers,)

    # step3: 目标函数和优化器
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr = lr, weigth_decay = opt.weight_decay)

    # step4：统计指标：平滑处理后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        pass

@t.no_grad()
def val(model, dataloader):

    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''

    pass

@t.no_grad()
def test(**kwargs):

    '''
    测试
    '''

    pass


def help():
    '''
    打印帮助信息
    '''
    print('help')

if __name__ == '__main__':

    import fire
    fire.Fire()

    '''
    通过python main.py <function> --args=xx的方式执行训练或测试
    '''