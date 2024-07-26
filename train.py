import argparse
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data as data
import os
from datasets import MYCIFAR10
from models import ResNet18, ResNet50
from utils import get_parameter_groups,boolean_string


#setting of hyperparas
def parse_args():
    parser = argparse.ArgumentParser(description = 'Training target models using PyTorch')
    parser.add_argument('--augmentations', default = True, type=boolean_string, help='Include data augmentations')
    #network
    parser.add_argument('--model', default = 'resnet50', type=str, help='Number of classes')
    parser.add_argument('--num_classes', default = 10, type=int, help='Number of classes')
    parser.add_argument('--activation', default ='relu', type=str, help='Activation function')

    # optimization:
    parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to be resumed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight momentum of SGD optimizer')
    parser.add_argument('--epochs', default='200', type=int, help='number of epochs')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')  # was 5e-4 for batch_size=128
    parser.add_argument('--num_workers', default = 2, type=int, help='Data loading threads')
    parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')

    # LR scheduler
    parser.add_argument('--lr_scheduler', default='reduce_on_plateau', type=str, help='reduce_on_plateau/multi_step')
    parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
    parser.add_argument('--patience', default=3, type=int, help='LR schedule patience for early stopping')
    parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')

    parser.add_argument('--checkpoint_dir', default='./results', type=str, help='Path of saved model')

    args = parser.parse_args()
    return args

def train(model, device, train_loader, optimizer, epoch):
    model.train() 
    train_loss = 0
    predicted = []
    labels = []
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):  # train a single step

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs['logits'], targets)
        loss.backward()
        optimizer.step()
        train_loss += loss
        _,preds = outputs['logits'].max(1)

        preds =preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        predicted.extend(preds)
        labels.extend(targets_np)

    N = batch_idx + 1
    train_loss = train_loss / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    print('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'.format(epoch, train_loss, train_acc))


def validate(model, device, val_loader, epoch, **kargs):
    global best_metric
    global best_epoch 

    model.eval()
    val_loss = 0
    predicted = []
    labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs['logits'], targets)
            val_loss += loss
            _,preds = outputs['logits'].max(1)

            preds = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            predicted.extend(preds)
            labels.extend(targets_np)
        
        N = batch_idx + 1
        val_loss    = val_loss / N
        predicted = np.asarray(predicted)
        val_acc = 100.0 * np.mean(predicted == labels)
        if kargs['metric'] == 'accuracy':
            metric = val_acc
        elif kargs['metric'] == 'loss':
            metric = val_loss
        else:
            raise AssertionError('Unknown metric for optimization {}'.format(kargs['metric']))
        
        if not os.path.exists(kargs['checkpoint_dir']):
            os.makedirs(kargs['checkpoint_dir'])

        #save model
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(kargs['checkpoint_dir'], 'ckpt_epoch_{}_{}_{:.2f}.pth'.format(epoch,kargs['metric'],metric)))

        if (epoch == 1):
            best_metric = metric
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(kargs['checkpoint_dir'], 'best_epoch_{}_{}_{:.2f}.pth'.format(epoch,kargs['metric'],metric)))
        else:
            if (kargs['metric'] == 'accuracy' and metric > best_metric) or (kargs['metric'] == 'loss' and metric < best_metric) :
                os.remove(os.path.join(kargs['checkpoint_dir'], 'best_epoch_{}_{}_{:.2f}.pth'.format(best_epoch, kargs['metric'],best_metric)))
                torch.save(model.state_dict(), os.path.join(kargs['checkpoint_dir'], 'best_epoch_{}_{}_{:.2f}.pth'.format(epoch,kargs['metric'],metric)))
                best_metric = metric
                best_epoch = epoch

        print('Epoch #{} (VAL): loss={}\tacc={:.2f}\tbest_metric({})={}'.format(epoch, val_loss, val_acc, kargs['metric'], best_metric))

        # updating learning rate if we see no improvement
        if kargs['early_stopping']:
            kargs['lr_scheduler'].step(metrics=metric)
        else:
            kargs['lr_scheduler'].step()

if __name__ == "__main__":
    #path
    train_path = 'data/train.txt'
    val_path = 'data/val.txt'
    #
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    #dataloader
    train_set = MYCIFAR10(train = True, list_path = train_path, augmentation = args.augmentations)
    val_set = MYCIFAR10(train = False, list_path = val_path)

    train_loader = data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True,
        num_workers = args.num_workers, pin_memory = device)
    
    val_loader = data.DataLoader(val_set, batch_size = args.batch_size, shuffle = False,
        num_workers = args.num_workers, pin_memory = device)
    #network
    strides = [1, 2, 2, 2]
    conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}

    if args.model =='resnet18':
        model = ResNet18(num_classes = args.num_classes, activation = args.activation,conv1 = conv1, strides = strides).to(device)
    if args.model =='resnet50':
        model = ResNet50(num_classes = args.num_classes, activation = args.activation,conv1 = conv1, strides = strides).to(device)   

    decay, no_decay = get_parameter_groups(model)

    optimizer = torch.optim.SGD([{'params': decay.values(), 'weight_decay': args.wd}, {'params': no_decay.values(), 'weight_decay': 0.0}],
                      lr=args.lr, momentum=args.momentum, nesterov=args.momentum > 0)
    
    if args.metric == 'accuracy':
        metric_mode = 'max'
    elif args.metric == 'loss':
        metric_mode = 'min'

    if args.lr_scheduler == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=  metric_mode,
            factor=args.factor,
            patience=args.patience,
            verbose=True,
            cooldown=args.cooldown
        )
    else:
        raise AssertionError('illegal LR scheduler {}'.format(args.lr_scheduler))

    for epoch in range(1, args.epochs + 1):
        train(model,device,train_loader,optimizer,epoch)
        validate(model,device, val_loader, epoch, metric = args.metric, lr_scheduler = lr_scheduler,
                  early_stopping = True, checkpoint_dir = args.checkpoint_dir)
        






