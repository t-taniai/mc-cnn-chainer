import chainer
import chainer.serializers

import glob
import chainer.links as L
import chainer.functions as F
import numpy as np
import stereo_utils as su

def get_arg(args):
    if hasattr(args, '__getitem__'):
        return args[0]
    return args

class MCCNN_pretrained(chainer.Chain):
    def __init__(self, filename:str=""):
        super(MCCNN_pretrained, self).__init__()
        with self.init_scope():
            self._layers1 = chainer.ChainList()
            self._layers2 = chainer.ChainList()
        
        if len(filename) > 0:
            print('--- Loading a MC-CNN pretrained model ---')
            files1W = glob.glob(filename + '_1_*W.bin')
            files1B = glob.glob(filename + '_1_*B.bin')
            self._add_convs(files1W, files1B, self._layers1)

            files2W = glob.glob(filename + '_2_*W.bin')
            files2B = glob.glob(filename + '_2_*B.bin')
            if files2W is not None and len(files2W) > 0:
                self._add_convs(files2W, files2B, self._layers2)

    def __call__(self, *args, **kwargs):
        with chainer.no_backprop_mode():
            ndisp = int(get_arg(args[2])) if len(args) >= 2 else 64
            g0 = self._to_gray(args[0])
            g1 = self._to_gray(args[1])

        x0 = self._extract_feature(g0)
        x1 = self._extract_feature(g1)

        # fast mode
        if len(self._layers2) == 0:
            x0 = F.normalize(x0)
            x1 = F.normalize(x1)
            #v = -su.reduce_to_vol(x0, x1, ndisp)
            v = -su.reduce_to_prodvol(x0, x1, ndisp)
            return v

        # accurate mode
        v = su.reduce_to_vol(x0, x1, ndisp,
            lambda y0, y1: self._looping(
                F.concat((y0, y1), 1), self._layers2, F.relu, F.sigmoid
            )
        )
        return v

    def _to_gray(self, x):
        if x.shape[1] == 3:
            g = 0.114 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.299 * x[:, 2, :, :]
        else:
            g = x[:, 0, :, :]

        return g[:, None]

    def _load_as_array(self, filename):
        buffer = np.fromfile(filename, np.byte).tobytes()
        dim = np.frombuffer(buffer, dtype=np.int32, count=1, offset=0)[0]
        shape = np.frombuffer(buffer, dtype=np.int32, count=dim, offset=4)
        array = np.frombuffer(buffer, dtype=np.float32, offset=4+4*dim)

        return np.reshape(array, shape)

    def _add_convs(self, filesW, filesB, layerlist):
        filesW.sort()
        filesB.sort()
        for i, fileW in enumerate(filesW):
            fileB = filesB[i]
            w = self._load_as_array(fileW)
            b = self._load_as_array(fileB)
            oc, ic, ky, kx = w.shape
            print(fileW + ' --- W:' + str(w.shape) + ' + b:' + str(b.shape))
            layerlist.add_link(
                L.Convolution2D(ic, out_channels=oc, ksize=(ky, kx), pad=(ky//2, kx//2), initialW=w, initial_bias=b)
            )
        print('')

    def _extract_feature(self, x_gray):
        with chainer.no_backprop_mode():
            mean = self.xp.mean(x_gray, (2, 3), keepdims=True)
            std = self.xp.std(x_gray, (2, 3), keepdims=True)
            x_gray = (x_gray - mean) / std

        return self._looping(
            chainer.Variable(x_gray),
            self._layers1,
            F.relu,
            F.identity if len(self._layers2) == 0 else F.relu
        )

    def _looping(self, x, layers, activ, lastactiv):
        n = len(layers)
        for i in range(n):
            x = lastactiv(layers[i](x)) if (i == n-1) else activ(layers[i](x))
        return x

