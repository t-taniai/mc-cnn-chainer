
from __future__ import print_function

import argparse
import chainer
import cv2
import numpy as np
import mcnet
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_image(file):
    img = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
    if img.ndim == 3:
        img = np.rollaxis(img, 2, 0)
    else:
        img = img[None]
    return img

def main():
    parser = argparse.ArgumentParser(description='Dynamic SGM Net')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='output', help='Directory to output the result')
    parser.add_argument('--vol', '-v', type=str2bool, default=False, help='Save cost volume data')
    args = parser.parse_args()
    outdir = args.out

    print('cuda:' + str(chainer.cuda.available))
    print('cudnn:' + str(chainer.cuda.cudnn_enabled))
    print('GPU: {}'.format(args.gpu))
    print('outdir: ', outdir)
    print('')

    chainer.config.train = False
    chainer.set_debug(False)
    chainer.using_config('use_cudnn', 'auto')

    # Load MC-CNN pre-trained models from
    # kitti_fast, kitti_slow, kitti2015_fast, kitti2015_slow, mb_fast, mb_slow
    model_kitti = mcnet.MCCNN_pretrained('mccnn/kitti_fast')
    model_mb = mcnet.MCCNN_pretrained('mccnn/mb_slow')

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_kitti.to_gpu()  # Copy the model to the GPU
        model_mb.to_gpu()  # Copy the model to the GPU

    samples = []
    #samples.append((model_mb, 'mb2014', 145))
    samples.append((model_kitti, 'kitti', 70))

    for sample in samples:
        model, target, ndisp = sample
        print('Processing ' + target)
        im0 = load_image(os.path.join('input', target, 'im0.png')).astype(np.float32)
        im1 = load_image(os.path.join('input', target, 'im1.png')).astype(np.float32)
        inputs = (im0, im1, np.array([ndisp]))

        batch = chainer.dataset.concat_examples([inputs], args.gpu)
        with chainer.no_backprop_mode():

            vol = model(*batch)[0].array
            disp = vol.argmin(0).astype(np.float32) * (255 / ndisp)

            os.makedirs(os.path.join(args.out, target), exist_ok=True)
            cv2.imwrite(os.path.join(args.out, target, 'disp0.png'), chainer.cuda.to_cpu(disp))

            if args.vol:
                vol.tofile(os.path.join(args.out, target, 'im0.bin'))
    

if __name__ == '__main__':
    main()

