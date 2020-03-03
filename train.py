import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.SSL_train import SSL_Train
from io_utils import model_dict, parse_args, get_resume_file  
import sys

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        print('lr:',param_group['lr'])
    if epoch%50 == 0:
        lr = 0.1*(0.1**(epoch//50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif optimization == 'Sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-3)
    else:
       raise ValueError('Unknown optimization')
    
    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        
        adjust_learning_rate(optimizer,epoch)
        model.train()
        model.train_loop(epoch, base_loader,  optimizer )
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : 
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')


    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 

    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224

    optimization = 'Sgd'

    if params.method in ['baseline', 'baseline_dist'] :
        base_datamgr = SimpleDataManager(image_size, batch_size = 200)
        base_loader = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr = SimpleDataManager(image_size, batch_size = 64)
        val_loader = val_datamgr.get_data_loader( val_file, aug = False)
        model = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')
    
    elif params.method in ['SSL'] :
        base_datamgr = SimpleDataManager(image_size, batch_size = 200)
        base_loader = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr = SimpleDataManager(image_size, batch_size = 64)
        val_loader = val_datamgr.get_data_loader( val_file, aug = False)
        model = SSL_Train( model_dict[params.model], params.num_classes)
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    #Prepare checkpoint_dir
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    print('checkpoint_dir',params.checkpoint_dir)

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
