import os
import sys
import numpy as np
from numpy.testing import assert_allclose
import time
import math

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import mxnet as mx
from symbol_resnet import get_symbol

def check_speed(sym, ctx, scale=1.0, N=100):
    exe = sym.simple_bind(grad_req='write', **ctx)
    init = [np.random.normal(size=arr.shape, scale=scale) for arr in exe.arg_arrays]
    for arr, iarr in zip(exe.arg_arrays, init):
        arr[:] = iarr.astype(arr.dtype)

    exe.forward(is_train=False)
    #exe.backward(exe.outputs[0])
    exe.outputs[0].wait_to_read()

    tic = time.time()
    for i in range(N):
        exe.forward(is_train=False)
        exe.backward(exe.outputs[0])
        exe.outputs[0].wait_to_read()
    return (time.time() - tic)*1000.0/N


sym_res = get_symbol()
print(mx.visualization.print_summary(sym_res, shape={'data': (1, 3, 224, 224)}))

#ctx_list = [{'ctx': mx.gpu(0), 'data': (1, 3, 224, 224)}]
#for ctx in ctx_list:
#    print(ctx, check_speed(sym_res, ctx, N=10))
