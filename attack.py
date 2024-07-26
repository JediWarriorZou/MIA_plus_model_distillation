import argparse
import torch.utils.data as data
import numpy as np
import time
import os
import torch
import torch.nn as nn
import logging

from utils import boolean_string,get_normalized_tensor,calc_acc_precision_recall,set_logger
from datasets import MYCIFAR10
from models import ResNet18

from art.attacks.inference.membership_inference import SelfInfluenceFunctionAttack
from art.estimators.classification import PyTorchClassifier

def parse_args():

    parser = argparse.ArgumentParser(description='Membership inference attack script')

    parser = argparse.ArgumentParser(description='Membership inference attack script')
    parser.add_argument('--checkpoint_dir', default='results/checkpoints/resnet18', type=str, help='checkpoint dir')
    parser.add_argument('--checkpoint_file', default='best_epoch_196_accuracy_86.12.pth', type=str, help='checkpoint path file name')
    parser.add_argument('--attack', default='self_influence', type=str, help='MI attack: gap/black_box/boundary_distance/self_influence')
    parser.add_argument('--attacker_knowledge', type=float, default=0.5, help='The portion of samples available to the attacker.')
    parser.add_argument('--output_dir', default='results', type=str, help='attack directory')
    parser.add_argument('--generate_mi_data', default=True, type=boolean_string, help='To generate MI data')
    parser.add_argument('--fast', default=False, type=boolean_string, help='Fast fit (500 samples) and inference (2500 samples)')

    #data and network
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--num_workers', default = 2, type=int, help='Data loading threads')
    parser.add_argument('--num_classes', default = 10, type=int, help='Number of classes')
    parser.add_argument('--activation', default ='relu', type=str, help='Activation function')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight momentum of SGD optimizer')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay') 

    # self_influence attack params
    parser.add_argument('--miscls_as_nm', default=True, type=boolean_string, help='Label misclassification is inferred as non members')
    parser.add_argument('--adaptive', default=True, type=boolean_string, help='Using train loader of influence function with augmentations, adaSIF method')
    parser.add_argument('--average', default=False, type=boolean_string, help='Using train loader of influence function with augmentations, ensemble method')
    parser.add_argument('--rec_dep', type=int, default=1, help='Recursion_depth of the influence functions.')
    parser.add_argument('--r', type=int, default=1, help='Number of iterations of which to take the avg of the Hessian_estimate calculation.')

    args = parser.parse_args()

    return args

def data_preparation(**kargs):
    member_path = 'data/member_1000.txt'
    non_member_path = 'data/non_member_1000.txt'
    data_dir = 'data'

    member_set = MYCIFAR10(train = False, list_path = member_path)
    non_member_set = MYCIFAR10(train = False, list_path = non_member_path)
    member_loader = data.DataLoader(member_set, batch_size = kargs['batch_size'], shuffle = False,
        num_workers = kargs['num_workers'], pin_memory = kargs['device'])
    non_member_loader = data.DataLoader(non_member_set, batch_size = kargs['batch_size'], shuffle = False,
        num_workers = kargs['num_workers'], pin_memory = kargs['device'])
    
    X_member = get_normalized_tensor(member_loader, kargs['img_shape'], kargs['batch_size'])
    Y_member = np.asarray(member_loader.dataset.labels)
    X_non_member = get_normalized_tensor(non_member_loader, kargs['img_shape'], kargs['batch_size'])
    Y_non_member = np.asarray(non_member_loader.dataset.labels)

    rand_gen = np.random.RandomState(int(time.time()))

    # building train/test set for members
    membership_train_size = int(kargs['attacker_knowledge'] * X_member.shape[0])
    membership_test_size = X_member.shape[0] - membership_train_size
    train_member_inds = rand_gen.choice(X_member.shape[0], membership_train_size, replace = False)
    train_member_inds.sort()
    X_member_train = X_member[train_member_inds]
    Y_member_train = Y_member[train_member_inds]

    test_member_inds = np.asarray([i for i in np.arange(X_member.shape[0]) if i not in train_member_inds])
    test_member_inds = rand_gen.choice(test_member_inds, membership_test_size, replace=False)
    test_member_inds.sort()
    X_member_test = X_member[test_member_inds]
    Y_member_test = Y_member[test_member_inds]

    # building train/test set for non members
    non_membership_train_size = membership_train_size
    non_membsership_test_size = membership_test_size
    train_non_member_inds = rand_gen.choice(X_non_member.shape[0], non_membership_train_size, replace=False)
    train_non_member_inds.sort()
    X_non_member_train = X_non_member[train_non_member_inds]
    Y_non_member_train = Y_non_member[train_non_member_inds]

    test_non_member_inds = np.asarray([i for i in np.arange(X_non_member.shape[0]) if i not in train_non_member_inds])
    test_non_member_inds = rand_gen.choice(test_non_member_inds, non_membsership_test_size, replace=False)
    test_non_member_inds.sort()
    X_non_member_test = X_non_member[test_non_member_inds]
    Y_non_member_test = Y_non_member[test_non_member_inds]

    np.save(os.path.join(data_dir, 'X_member_train.npy'), X_member_train)
    np.save(os.path.join(data_dir, 'y_member_train.npy'), Y_member_train)
    np.save(os.path.join(data_dir, 'X_non_member_train.npy'), X_non_member_train)
    np.save(os.path.join(data_dir, 'y_non_member_train.npy'), Y_non_member_train)
    np.save(os.path.join(data_dir, 'X_member_test.npy'), X_member_test)
    np.save(os.path.join(data_dir, 'y_member_test.npy'), Y_member_test)
    np.save(os.path.join(data_dir, 'X_non_member_test.npy'), X_non_member_test)
    np.save(os.path.join(data_dir, 'y_non_member_test.npy'), Y_non_member_test)
    print("Data preparation is finished.")


if __name__ == "__main__":
    data_dir = 'data'
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    img_shape = [32, 32, 3]

    log_file = os.path.join(args.output_dir, 'log.log')
    set_logger(log_file)
    logger = logging.getLogger()

    data_preparation(batch_size = args.batch_size, num_workers = args.num_workers, 
                     device = device,attacker_knowledge = args.attacker_knowledge, img_shape = img_shape)
    
    logger.info('loading data..')
    
    X_member_train = np.load(os.path.join(data_dir, 'X_member_train.npy'))
    y_member_train = np.load(os.path.join(data_dir, 'y_member_train.npy'))
    X_non_member_train = np.load(os.path.join(data_dir, 'X_non_member_train.npy'))
    y_non_member_train = np.load(os.path.join(data_dir, 'y_non_member_train.npy'))
    X_member_test = np.load(os.path.join(data_dir, 'X_member_test.npy'))
    y_member_test = np.load(os.path.join(data_dir, 'y_member_test.npy'))
    X_non_member_test = np.load(os.path.join(data_dir, 'X_non_member_test.npy'))
    y_non_member_test = np.load(os.path.join(data_dir, 'y_non_member_test.npy'))

    #network
    strides = [1, 2, 2, 2]
    conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}

    logger.info('==> Building model..')
    
    model = ResNet18(num_classes = args.num_classes, activation = args.activation,conv1 = conv1, strides = strides).to(device)
    model_state = torch.load(best_checkpoint, map_location=torch.device(device))
    model.load_state_dict(model_state)
    model.eval()

    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd,
    nesterov=args.momentum > 0)

    loss = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss, optimizer=optimizer,
                                input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=args.num_classes)
    
    attack = SelfInfluenceFunctionAttack(classifier, debug_dir = args.output_dir, miscls_as_nm=args.miscls_as_nm,
                                         adaptive=args.adaptive, average=args.average, for_ref=False,
                                         rec_dep=args.rec_dep, r=args.r)
    
    print('Begin fitting.')
    logger.info('Fitting {} attack...'.format(args.attack))
    start = time.time()
    attack.fit(x_member=X_member_train, y_member=y_member_train,
               x_non_member=X_non_member_train, y_non_member=y_non_member_train)
    logger.info('Fitting time: {} sec'.format(time.time() - start))
    
    print('Begin inference.')
    start = time.time()
    inferred_member = attack.infer(X_member_test, y_member_test, **{'infer_set': 'member_test'})
    inferred_non_member = attack.infer(X_non_member_test, y_non_member_test, **{'infer_set': 'non_member_test'})
    logger.info('Inference time: {} sec'.format(time.time() - start))
    print('Compute metrics.')

    calc_acc_precision_recall(inferred_non_member, inferred_member)

 