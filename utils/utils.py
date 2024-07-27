import torch.nn as nn
import torch
import numpy as np
import logging
import sys
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support
from torchvision import transforms
from models import ResNet18, ResNet50,AlexNetCIFAR

def get_parameter_groups(net: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    no_decay = dict()
    decay = dict()
    for name, m in net.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            decay[name + '.weight'] = m.weight
            decay[name + '.bias'] = m.bias
        elif isinstance(m, nn.BatchNorm2d):
            no_decay[name + '.weight'] = m.weight
            no_decay[name + '.bias'] = m.bias
        else:
            if hasattr(m, 'weight'):
                no_decay[name + '.weight'] = m.weight
            if hasattr(m, 'bias'):
                no_decay[name + '.bias'] = m.weight

    # remove all None values:
    del_items = []
    for d, v in decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        decay.pop(d)

    del_items = []
    for d, v in no_decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        no_decay.pop(d)

    return decay, no_decay

def boolean_string(s):
    # to use --use_bn True or --use_bn False in the shell. See:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_normalized_tensor(loader: torch.utils.data.DataLoader, img_shape, batch_size=None):
    """ Returning a normalized tensor"""
    if batch_size is None:
        batch_size = loader.batch_size
    size = len(loader.dataset)
    X = -1.0 * np.ones(shape=(size, img_shape[2], img_shape[0], img_shape[1]), dtype=np.float32)
    for batch_idx, (inputs, targets) in enumerate(loader):
        b = batch_idx * batch_size
        e = b + targets.shape[0]
        X[b:e] = inputs.cpu().numpy()

    return X

def calc_acc_precision_recall(inferred_non_member, inferred_member,logger):
    
    member_acc = np.mean(inferred_member == 1)
    non_member_acc = np.mean(inferred_non_member == 0)
    acc = (member_acc * len(inferred_member) + non_member_acc * len(inferred_non_member)) / (len(inferred_member) + len(inferred_non_member))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(
        y_true=np.concatenate((np.zeros(len(inferred_non_member)), np.ones(len(inferred_member)))),
        y_pred=np.concatenate((inferred_non_member, inferred_member)),
    )
    logger.info('member acc: {}, non-member acc: {}, balanced acc: {}, precision/recall(member): {}/{}, precision/recall(non-member): {}/{}'
                .format(member_acc, non_member_acc, acc, precision[1], recall[1], precision[0], recall[0]))
    
def normalize(x, rgb_mean, rgb_std):
    """
    :param x: np.ndaaray of image RGB of (3, W, H), normalized between [0,1]
    :param rgb_mean: Tuple of (RED mean, GREEN mean, BLUE mean)
    :param rgb_std: Tuple of (RED std, GREEN std, BLUE std)
    :return np.ndarray transformed by x = (x-mean)/std
    """
    transform = transforms.Normalize(rgb_mean, rgb_std)
    x_tensor = torch.tensor(x)
    x_new = transform(x_tensor)
    x_new = x_new.cpu().numpy()
    return x_new

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W) or (feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) or (H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    if len(x.shape) == 3:
        X = np.transpose(X, [1, 2, 0])
    else:
        X = np.transpose(X, [0, 2, 3, 1])
    return X

def set_logger(log_file):
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)]
                        )
    
