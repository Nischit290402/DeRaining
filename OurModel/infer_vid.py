import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from netsV2 import DGNLNet
from config import test_raincityscapes_path
from misc import check_mkdir

from video_extr import breakVideo
from video_gen import makeVideo

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'DGNLNet'
args = {
    'snapshot': '40000_orig',
    'depth_snapshot': ''
}

transform = transforms.Compose([
    transforms.Resize([512,1024]),
    transforms.ToTensor() ])

# root = os.path.join(test_raincityscapes_path, 'test')
root = 'E:/Minor_Project/Datasets/sampleRain.mp4'
tempIn = 'E:/Minor_Project/Datasets/tempIn/'
tempOut = os.path.join(ckpt_path, exp_name, '(%s) prediction_%s_vid' % (
                    exp_name, args['snapshot'][:5]))
to_pil = transforms.ToPILImage()

print("Breaking video into frames...")
breakVideo(root, tempIn)

print("Processing frames...")
def main():
    net = DGNLNet().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    avg_time = 0

    with torch.no_grad():
        img_list = [img_name for img_name in os.listdir(tempIn)]
        
        for idx, img_name in enumerate(img_list):
            check_mkdir(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s_vid' % (exp_name, args['snapshot'][:5])))
            if len(args['depth_snapshot']) > 0:
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['depth_snapshot'])))

            img = Image.open(os.path.join(tempIn, img_name)).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).cuda()

            start_time = time.time()

            #res, dps = net(img_var)
            res = net(img_var)

            torch.cuda.synchronize()

            # if len(args['depth_snapshot']) > 0:
            #     depth_res = depth_net(res, depth_optimize = True)

            avg_time = avg_time + time.time() - start_time

            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), avg_time/(idx+1)))

            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))


            result.save(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s_vid' % (
                    exp_name, args['snapshot'][:5]), img_name))


            # if len(args['depth_snapshot']) > 0:
            #     depth_result = transforms.Resize((h, w))(to_pil(dps.data.squeeze(0).cpu()))
            #     depth_result.save(
            #         os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
            #             exp_name, args['depth_snapshot']), img_name))

    print("Generating video from frames...")
    makeVideo(tempOut, 'E:/Minor_Project/Datasets/sampleRainOutV2.mp4')

    # Delete tempIn folder
    for file in os.listdir(tempIn):
        os.remove(os.path.join(tempIn, file))
    os.rmdir(tempIn)

    # Delete tempOut folder
    # for file in os.listdir(tempOut):
    #     os.remove(os.path.join(tempOut, file))
    # os.rmdir(tempOut)

if __name__ == '__main__':
    main()
