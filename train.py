from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import util
from util import utils
import lr_scheduler as L
from models import *
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
import random
from util.nlp_utils import *
from Data import *

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-task', default='ag', type=str, choices=['ag', 'amazon'])
    parser.add_argument('-model', default='h_attention', type=str,
                        choices=['slstm', 'glstm', 'h_attention', 'hglstm'])
    parser.add_argument('-use_depparse', default=False, action='store_true')
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = util.utils.read_config(opt.task + '_config.yaml')
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model, vocab, dataloader, scheduler, optim, updates):
    scores = []
    max_acc = 0.
    for epoch in range(1, config.epoch + 1):
        total_right = 0
        total_num = 0
        total_loss = 0.
        start_time = time.time()

        if config.schedule:
            scheduler.step()
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        model.train()

        train_data = dataloader.train_batches
        random.shuffle(train_data)
        for batch in tqdm(train_data, disable=not args.verbose):
            model.zero_grad()
            outputs = model(batch, use_cuda)
            label = batch.label
            if use_cuda:
                label = label.cuda()
            right_num = torch.sum(outputs.max(-1)[1] == label).item()
            total_right += right_num
            total_num += batch.batch_size
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            total_loss += loss.data.item()

            optim.step()
            updates += 1  # 进行了一次更新

            # 多少次更新之后记录一次
            if updates % config.eval_interval == 0 or args.debug:
                # logging中记录的是每次更新时的epoch，time，updates，correct等基本信息.
                # 还有score分数的信息
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss / config.eval_interval,
                           total_right / float(total_num)))
                print('evaluating after %d updates...\r' % updates)
                score = eval(model, dataloader, epoch, updates, do_test=False)
                scores.append(score)
                if score >= max_acc:
                    save_model(log_path + str(score) + '_checkpoint.pt', model, optim, updates)
                    max_acc = score

                model.train()
                total_loss = 0.
                total_right = 0
                total_num = 0
                start_time = time.time()

            if updates % config.save_interval == 0:  # 多少次更新后进行保存一次
                save_model(log_path + str(updates) + '_updates_checkpoint.pt', model, optim, updates)
    model = load_model(log_path + str(max_acc) + '_checkpoint.pt', model)
    test_acc = eval(model, dataloader, -1, -1, True)
    return max_acc, test_acc


def eval(model, dataloader, epoch, updates, do_test=False):
    model.eval()
    total_right = 0
    total_num = 0
    if do_test:
        data_batches = dataloader.test_batches
    else:
        data_batches = dataloader.dev_batches
    for batch in tqdm(data_batches, disable=not args.verbose):
        label = batch.label
        if use_cuda:
            label = label.cuda()
        outputs = model(batch, use_cuda)
        right_num = torch.sum(outputs.max(-1)[1] == label).item()
        total_right += right_num
        total_num += batch.batch_size
    acc = total_right / float(total_num)
    logging_csv([epoch, updates, acc])
    print(acc, flush=True)
    return acc


def save_model(path, model, optim, updates):
    '''保存的模型是一个字典的形式, 有model, config, optim, updates.'''

    # 如果使用并行的话使用的是model.module.state_dict()
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def load_model(path, model):
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints['model'])
    return model


def main():
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # checkpoint
    if args.restore:  # 存储已有模型的路径
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))

    # word2id, id2word, word2count = load_vocab(args.vocab_file, args.vocab_size)
    vocab = Vocab(config.vocab, config.data, config.vocab_size)

    # Load data
    start_time = time.time()
    dataloader = DataLoader(config, args.task, config.has_dev, config.batch_size, vocab, args.model, args.use_depparse,
                            args.notrain, args.debug)
    print("DATA loaded!")

    torch.backends.cudnn.benchmark = True

    # data
    print('loading data...\n')
    print('loading time cost: %.3f' % (time.time() - start_time))

    # model
    print('building model...\n')
    # configure the model
    # Model and optimizer
    if args.model == 'h_attention':
        model = hierarchical_attention(config, vocab, use_cuda)
    elif args.model == 'slstm':
        model = SLSTM(config, vocab, use_cuda)
    elif args.model == 'glstm':
        model = GLSTM(config, vocab, use_cuda)
    elif args.model == 'hglstm':
        model = HGLSTM(config, vocab, use_cuda)
    if args.restore:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()
        # lm_model.cuda()
    if len(args.gpus) > 1:  # 并行
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
    logging(repr(model) + "\n\n")  # 记录这个文件的框架

    # total number of parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]

    logging('total number of parameters: %d\n\n' % param_count)

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
        ori_updates = updates
    else:
        updates = 0

    # optimizer
    if args.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                      lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

    # if opt.pretrain:
    # pretrain_lm(lm_model, vocab)
    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    else:
        scheduler = None

    if not args.notrain:
        max_acc, test_acc = train(model, vocab, dataloader, scheduler, optim, updates)
        logging("Best accuracy: %.2f, test accuracy: %.2f\n" % (max_acc*100, test_acc*100))
    else:
        assert args.restore is not None
        eval(model, vocab, dataloader, 0, updates, do_test=True)


if __name__ == '__main__':
    main()
