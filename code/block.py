from collections import OrderedDict
import torch
import torch.nn as nn

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'selu':
        layer = nn.SELU()
    elif act_type == 'elu':
        layer = nn.ELU()
    elif act_type == 'silu':
        layer = nn.SiLU()
    elif act_type == 'rrelu':
        layer = nn.RReLU()
    elif act_type == 'celu':
        layer = nn.CELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='elu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='elu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv1 = conv_block(in_nc, 64, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    conv2 = conv_block(64, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv1,conv2)


class CA(nn.Module):

    def __init__(self, nf):
        super(CA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv0 = conv_block(nf, nf, kernel_size=3, stride=1, act_type='elu')
        self.conv1 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu',groups = nf)
        self.conv2 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu',groups = nf)
        self.conv3 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu')

    def forward(self, x):
        x1 = self.gap(x)
        #t=x
        #b=torch.std(t.view(t.size(0),t.size(1),-1),2)
        #x2=b.view(b.size(0),b.size(1),1,1)
        #feat = torch.cat((x1,x2),1)
        feat=x1
        w = self.conv3(self.conv2(self.conv1(self.conv0(feat))))
        out = x.mul(w)
        return out


class Residual(nn.Module):

    def __init__(self, nf,act_type='elu'):
        super(Residual,self).__init__()
        self.conv1 = conv_block(nf, nf, kernel_size=5, norm_type='batch', act_type=act_type)
        self.conv2 = conv_block(nf, nf, kernel_size=5, norm_type='batch',act_type=act_type,groups = nf)
        self.conv3 = conv_block(2*nf, nf, kernel_size=5, norm_type='batch',act_type=act_type,groups = nf)
        self.conv4 = conv_block(nf, nf, kernel_size=5, norm_type='batch',act_type=act_type)
        #self.conv41 = conv_block(nf, nf, kernel_size=5, norm_type='batch',act_type='selu')

        self.conv5 = conv_block(nf, nf, kernel_size=3, norm_type='batch', act_type=act_type)
        self.conv6 = conv_block(nf, nf, kernel_size=3,  norm_type='batch',act_type=act_type,groups = nf)
        self.conv7 = conv_block(2*nf, nf, kernel_size=3,  norm_type='batch', act_type=act_type,groups = nf)
        self.conv8 = conv_block(nf, nf, kernel_size=3, norm_type='batch',act_type=act_type)
        #self.conv81 = conv_block(nf, nf, kernel_size=3, norm_type='batch',act_type='selu')

        self.conv9 = conv_block(nf, nf, kernel_size=1, norm_type='batch', act_type=act_type)
        self.conv10 = conv_block(nf, nf, kernel_size=1, norm_type='batch',act_type=act_type,groups = nf)
        self.conv11 = conv_block(2*nf, nf, kernel_size=1, norm_type='batch', act_type=act_type,groups = nf)
        self.conv12 = conv_block(nf, nf, kernel_size=1, norm_type='batch',act_type=act_type)
        #self.conv121 = conv_block(nf, nf, kernel_size=1, norm_type='batch',act_type='selu')

        '''self.conv13 = conv_block(nf, nf, kernel_size=7, norm_type='batch', act_type='selu')
                                self.conv14 = conv_block(nf, nf, kernel_size=7, norm_type='batch',act_type='selu',groups = nf)
                                self.conv15 = conv_block(nf, nf, kernel_size=7, norm_type='batch',act_type='selu')'''
        self.c0 = conv_block(7*nf,nf, kernel_size=3)
        #self.w = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x1 = self.conv2(self.conv1(x))
        x1_1 = torch.cat((x,x1),1)
        x1_2 = self.conv4(self.conv3(x1_1))
        x1_3 = torch.cat((x1,x1_2),1)

        x2 = self.conv6(self.conv5(x))
        x2_1 = torch.cat((x,x2),1)
        x2_2 = self.conv8(self.conv7(x2_1))
        x2_3 = torch.cat((x2,x2_2),1)

        x3 = self.conv10(self.conv9(x))
        x3_1 = torch.cat((x,x3),1)
        x3_2 = self.conv12(self.conv11(x3_1))
        x3_3 = torch.cat((x3,x3_2),1)
        #x4 = self.conv15(self.conv14(self.conv13(x)))
        #out = x.mul(self.w) + x1
        out = torch.cat((x,x1_3,x2_3,x3_3),1)
        out = self.c0(out)
        return out



class MyNetwork(nn.Module):
    def  __init__(self, nf):
        super(MyNetwork, self).__init__()
        in_nc=3
        out_nc=3
        self.nf = nf
        n1 = nf
        #low frequecy features
        self.c0 = conv_block(in_nc, 16, kernel_size=7, act_type='elu')
        self.c1 = conv_block(16, 16, kernel_size=5, act_type='elu')
        self.g1 = conv_block(16, 32, kernel_size=3, stride=1, act_type='elu',groups = 16)
        self.g2 = conv_block(32, 32, kernel_size=5, act_type='elu',groups = 32)
        self.g3 = conv_block(32, nf, kernel_size=1, stride=1, act_type='elu')

        #grl

        self.grl1 = upconv_blcok(in_nc, 64, upscale_factor=4)
        self.grl2 = sequential(Residual(n1),CA(n1))
        self.grl3 = sequential(Residual(n1),CA(n1))
        self.grl4 = conv_block(n1, 32, kernel_size=3, stride=1, act_type='elu')
        self.grl5 = conv_block(32, 16, kernel_size=3, stride=1, act_type='elu')
        self.grl6 = conv_block(16, 3, kernel_size=3, stride=1, act_type='elu')


        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        #self.g1 = conv_block(nf, 16, kernel_size=1, stride=1, act_type='selu')
        #self.g2 = conv_block(16, nf, kernel_size=1, stride=1, act_type='sigmoid')
        #high frequecy features
        
        #self.i = nn.InstanceNorm2d(n1)
        self.r1 = sequential(Residual(n1),CA(n1))
        self.r2 = sequential(Residual(n1),CA(n1))
        self.r22 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu')
        self.r1_1 = conv_block(nf*2, nf, kernel_size=1, stride=1, act_type='elu')
        self.r3 = sequential(Residual(n1),CA(n1))
        self.r4 = sequential(Residual(n1),CA(n1))
        self.r42 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu')
        self.r1_2 = conv_block(nf*2, nf, kernel_size=1, stride=1, act_type='elu') 
        self.r5 = sequential(Residual(n1),CA(n1))
        self.r5_1 = sequential(Residual(n1),CA(n1))
        self.r5_2 = conv_block(nf, nf, kernel_size=1, stride=1, act_type='elu')
        self.r5_3 = conv_block(nf*2, nf, kernel_size=1, stride=1, act_type='elu')      
        self.u1 = upconv_blcok(nf, n1, upscale_factor=4)
        
        #resonstruction
        self.c2 = conv_block(n1,n1,kernel_size=1,act_type='elu')

        self.c4 = conv_block(nf*2,32,kernel_size=1,act_type='elu')
        self.c5 = conv_block(32*2,16,kernel_size=1,act_type='elu')
        self.c6 = conv_block(16*2,out_nc,kernel_size=1,act_type='elu')

        self.u2 = upconv_blcok(16, 16, upscale_factor=4)
        self.u2_1 = sequential(Residual(16),CA(16))
        #self.grl3 = conv_block(n1, 32, kernel_size=3, stride=1, act_type='elu')
        #self.grl4 = conv_block(32, 16, kernel_size=3, stride=1, act_type='elu')
        self.u2_2 = conv_block(16, 16, kernel_size=3, stride=1, act_type='elu')

        self.u3 = upconv_blcok(32, 32, upscale_factor=4)
        self.u3_1 = sequential(Residual(32),CA(32))
        #self.grl3 = conv_block(n1, 32, kernel_size=3, stride=1, act_type='elu')
        #self.grl4 = conv_block(32, 16, kernel_size=3, stride=1, act_type='elu')
        self.u3_2 = conv_block(32, 32, kernel_size=3, stride=1, act_type='elu')

        self.u4 = upconv_blcok(64, n1, upscale_factor=4)
        self.u4_1 = sequential(Residual(64),CA(64))
        #self.grl3 = conv_block(n1, 32, kernel_size=3, stride=1, act_type='elu')
        #self.grl4 = conv_block(32, 16, kernel_size=3, stride=1, act_type='elu')
        self.u4_2 = conv_block(64, 64, kernel_size=3, stride=1, act_type='elu')

    def forward(self, x):
        #grl
        w1 = self.grl6(self.grl5(self.grl4(self.grl3(self.grl2(self.grl1(x))))))
        
        #x1 = x1.mul(w)
        #low freq. features
        w0 = self.c1(self.c0(x))
        w00 =self.g1(w0)
        w =self.g3(self.g2(w00))

        #high freq. features

        x2 = self.r2(self.r1(w))
        x22 = self.r22(x2)
        x22 = torch.cat((w,x2),1)
        x22 = self.r1_1(x22)
        x3 = self.r4(self.r3(x22))
        x33 = self.r42(x3)
        x33 = torch.cat((w,x3),1)
        x33 = self.r1_2(x33)
        x4 = self.r5_1(self.r5(x33))
        x44 = self.r5_2(x4)
        x44 = torch.cat((w,x4),1)
        x44 = self.r5_3(x44)
        x5 = self.u1(x44)
        
        # reconstruction
        x1 = self.c2(x5)
        w = self.u4_2(self.u4_1(self.u4(w)))
        x1_1 = torch.cat((w,x1),1)

        x2 = self.c4(x1_1)
        w00= self.u3_2(self.u3_1(self.u3(w00)))
        x2_1 = torch.cat((w00,x2),1)

        x3 = self.c5(x2_1)
        w0= self.u2_2(self.u2_1(self.u2(w0)))
        x3_1 = torch.cat((w0,x3),1)

        x6 = self.c6(x3_1)

        out = w1+x6
        #out = self.c5(self.c4(final_d))
        return out


