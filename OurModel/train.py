import datetime
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import triple_transforms
from netsV2 import DGNLNet
from config import train_raincityscapes_path, test_raincityscapes_path
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir
from tqdm import tqdm

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'DGNLNet'

args = {
    'iter_num': 16,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'val_freq': 50000000,
    'img_size_h': 512,
	'img_size_w': 1024,
	'crop_size': 512,
    'snapshot_epochs': 16
}

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    #triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])


train_set = ImageFolder(train_raincityscapes_path, transform=transform, target_transform=transform, triple_transform=triple_transform, is_train=True)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
test1_set = ImageFolder(test_raincityscapes_path, transform=transform, target_transform=transform, is_train=False)
test1_loader = DataLoader(test1_set, batch_size=2)

criterion = DiceLoss()
criterion_depth = DiceLoss()
# criterion = nn.L1Loss()
# criterion_depth = nn.L1Loss()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    # print(train_raincityscapes_path)
    net = DGNLNet().cuda().train()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    # open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    epoch_num = args['iter_num']
    for epoch in range(epoch_num):
        train_loss_record = AvgMeter()
        train_net_loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()
        tqdm_bar = tqdm(train_loader, desc='Epoch {}'.format(epoch+1), unit='batch', bar_format='{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for inputs, gts, dps in tqdm_bar:
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(epoch) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(epoch) / args['iter_num']
                                                            ) ** args['lr_decay']
            
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            optimizer.zero_grad()

            result, depth_pred = net(inputs)

            loss_net = criterion(result, gts)
            loss_depth = criterion_depth(depth_pred, dps)

            loss = loss_net + loss_depth

            loss.backward()

            optimizer.step()

            tqdm_bar.set_postfix({'Loss': '{:.4f}'.format(loss.item())})
            tqdm_bar.update()

            torch.cuda.empty_cache()

            # for n, p in net.named_parameters():
            #     if n[-5:] == 'alpha':
            #         print(p.grad.data)
            #         print(p.data)

            train_loss_record.update(loss.data, batch_size)
            train_net_loss_record.update(loss_net.data, batch_size)
            train_depth_loss_record.update(loss_depth.data, batch_size)

            # epoch += 1

            log = '[iter %d], [train loss %.5f], [lr %.13f], [loss_net %.5f], [loss_depth %.5f]' % \
                  (epoch, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_net_loss_record.avg, train_depth_loss_record.avg)
            # print(log)
            # open(log_path, 'a').write(log + '\n')

            # if (epoch + 1) % args['val_freq'] == 0:
            #     validate(net, epoch, optimizer)

        if (epoch + 1) % args['snapshot_epochs'] == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (epoch + 1) )))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (epoch + 1) )))
            print('save model: %d.pth' % (epoch + 1))

def validate(net, epoch, optimizer):
    print('validating...')
    net.eval()

    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    iter_num1 = len(test1_loader)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test1_loader)):
            inputs, gts, dps = data
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            res = net(inputs)

            loss = criterion(res, gts)
            loss_record1.update(loss.data, inputs.size(0))

            print('processed test1 %d / %d' % (i + 1, iter_num1))


    snapshot_name = 'iter_%d_loss1_%.5f_loss2_%.5f_lr_%.6f' % (epoch + 1, loss_record1.avg, loss_record2.avg,
                                                               optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss1 %.5f], [loss2 %.5f]' % (epoch + 1, loss_record1.avg, loss_record2.avg))
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    main()
