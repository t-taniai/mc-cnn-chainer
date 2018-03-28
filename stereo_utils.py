import chainer
import chainer.functions as F
import numpy as np

class ProductVolumeFunc(chainer.Function):
    def __init__(self, ndisp):
        self.ndisp = ndisp

    def forward_cpu(self, inputs):
        l, r = inputs
        self.retain_inputs((0, 1))
        self.retain_outputs(())
        xp = chainer.cuda.get_array_module(l)

        n, c, h, w0 = l.shape
        w0 = l.shape[-1]
        w1 = r.shape[-1]
        d = self.ndisp

        v = xp.empty((n, d, h, w0), dtype=l.dtype)
        for d in range(0, d):
            x0 = np.maximum(0, w0 - w1 + d)
            x1 = np.maximum(0, w1 - w0 - d)

            ld = l[:, :, :, x0:]
            rd = r[:, :, :, x1:-d] if d > 0 else r[:, :, :, x1:]
            score = xp.sum(ld * rd, 1)
            p = w0 - score.shape[-1]
            if p > 0:
                pad = xp.broadcast_to(score[:, :, :1], score.shape[:-1] + (p,))
                score = xp.concatenate((pad, score), -1)
            v[:, d, :, :] = score
        return v,

    def forward_gpu(self, inputs):
        l, r = inputs
        self.retain_inputs((0, 1))
        self.retain_outputs(())

        n, c, h, w0 = l.shape
        w1 = r.shape[-1]
        d = self.ndisp

        v = chainer.cuda.cupy.empty((n, d, h, w0), dtype=l.dtype)
        chainer.cuda.elementwise(
            'raw T l, raw T r, int32 c, int32 h, int32 w0, int32 w1, int32 d',
            'T v',
            '''
               int n0 = i / (d * h * w0);
               int d0 = i / (h * w0) % d;
               int y0 = i / w0 % h;
               int x0 = i % w0;
               int x1 = x0 + w1 - w0 - d0;
               int shift = max(-x1, 0);
               x0 += shift;
               x1 += shift;
               
               T sum = 0;
               int j0 = x0 + w0 * (y0 + h * (c * n0));
               int j1 = x1 + w1 * (y0 + h * (c * n0));
               int s0 = h * w0;
               int s1 = h * w1;
               for (int i = 0; i < c; i++){
                 sum += l[j0] * r[j1];
                 j0 += s0;
                 j1 += s1;
               }
               v = sum;
            ''',
            'product_volume_fw')(l.reduced_view(), r.reduced_view(), c, h, w0, w1, d, v.reduced_view())
        return v,

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError()
        # Not checked correctness

        l, r = inputs
        gv = grad_outputs[0]
        xp = chainer.cuda.get_array_module(gv)

        n, c, h, w0 = l.shape
        w1 = r.shape[-1]
        d = self.ndisp

        l, r = inputs
        gv = grad_outputs[0]

        w0 = l.shape[-1]
        w1 = r.shape[-1]

        gl = xp.zeros_like(l)
        gr = xp.zeros_like(r)

        for d in range(0, self.ndisp):
            x0 = xp.maximum(0, w0 - w1 + d)
            x1 = xp.maximum(0, w1 - w0 - d)
            score = gv[:, d, :, :]
            if x0 > 0:
                score[:, :, x0] += xp.sum(score[:, :, :x0], -1)
                score = score[:, :, x0:]
            score = score[:, None]
            gr[:, :, :, x0:] += l[:, :, :, x0:] * score
            gl[:, :, :, x1:(w1-d)] += r[:, :, :, x1:(w1-d)] * score

        return gl, gr

def reduce_to_prodvol(l, r, ndisp):
    return ProductVolumeFunc(ndisp)(l, r)

def dot(x, y, axis=1):
    return F.sum(x * y, axis=axis, keepdims=True)

def reduce_to_vol(l, r, ndisp, reduce_func=dot):
    w0 = l.shape[-1]
    w1 = r.shape[-1]

    slices = []
    slices.append(reduce_func(l, r[:, :, :, -w0:]))
    for d in range(1, ndisp):
        x0 = np.maximum(0, w0 - w1 + d)
        x1 = np.maximum(0, w1 - w0 - d)
        score = reduce_func(l[:, :, :, x0:], r[:, :, :, x1:-d])
        #print(x0, x1, score.shape[-1])
        p = w0 - score.shape[-1]
        if p > 0:
            pad = F.broadcast_to(score[:, :, :, :1], score.shape[:-1] + (p, ))
            slices.append(F.concat((pad, score), -1))
        else:
            slices.append(score)

    vol_c = F.concat(slices, 1)

    return vol_c
