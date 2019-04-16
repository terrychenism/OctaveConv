import numpy as np
import mxnet as mx
from symbol_octConv import *
from symbol_basic import *

G   = 1
alpha=0.25
use_fp16=True
k_sec  = {  2:  3, \
            3:  4, \
            4:  6, \
            5:  3  }

def get_before_pool():
    data = mx.symbol.Variable(name="data")
    data = mx.sym.Cast(data=data, dtype=np.float16) if use_fp16 else data

    # conv1
    conv1 = Conv_BN_AC(data=data, num_filter=64,  kernel=(7,7), name='conv1', pad=(3,3), stride=(2,2))
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    num_in =  32
    num_mid = 64
    num_out = 256
    i = 1
    hf_conv2_x, lf_conv2_x = Residual_Unit_first( 
                             data = pool1,
                             alpha=alpha,
                             num_in = (num_in if i == 1 else num_out),   
                             num_mid = num_mid,
                             num_out = num_out, 
                             name = ('conv2_B%02d'% i),
                             first_block = (i==1), 
                             stride = ((1, 1) if (i == 1) else (1,1)) )


    for i in range(2, k_sec[2]+1):
        hf_conv2_x, lf_conv2_x = Residual_Unit( 
                                 hf_data = (hf_conv1_x if i == 1 else hf_conv2_x),
                                 lf_data = (lf_conv1_x if i == 1 else lf_conv2_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv2_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((1, 1) if (i == 1) else (1,1)) )

    # conv3
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    for i in range(1, k_sec[3]+1):
        hf_conv3_x, lf_conv3_x = Residual_Unit( 
                                 hf_data = (hf_conv2_x if i == 1 else hf_conv3_x),
                                 lf_data = (lf_conv2_x if i == 1 else lf_conv3_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv3_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)) )


    # conv4
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    for i in range(1, k_sec[4]+1):
        hf_conv4_x, lf_conv4_x = Residual_Unit( 
                                 hf_data = (hf_conv3_x if i == 1 else hf_conv4_x),
                                 lf_data = (lf_conv3_x if i == 1 else lf_conv4_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv4_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)) )


    # conv5
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    i = 1
    conv5_x = Residual_Unit_last( 
                             hf_data = (hf_conv4_x if i == 1 else hf_conv5_x),
                             lf_data = (lf_conv4_x if i == 1 else lf_conv5_x),
                             alpha=alpha,
                             num_in = (num_in if i == 1 else num_out),   
                             num_mid = num_mid,
                             num_out = num_out, 
                             name = ('conv5_B%02d'% i),
                             first_block = (i==1), 
                             stride = ((2, 2) if (i == 1) else (1,1)) )

    for i in range(2, k_sec[5]+1):
        conv5_x = Residual_Unit_norm( data = (conv4_x if i == 1 else conv5_x),
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv5_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)) )

    output = mx.sym.Cast(data=conv5_x, dtype=np.float32) if use_fp16 else conv5_x
    # output
    return output

def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="global-pool")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='classifier')
    return fc6

def get_symbol(num_classes = 1000):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out




