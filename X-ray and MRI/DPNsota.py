
import torch
import torch.nn as nn
import torch.nn.functional as F

from BioInspired import DCP_gray, DCP

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class Mdense_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Mdense_conv, self).__init__()
        self.conv = nn.Sequential(
            # nn.InstanceNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Pdense_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Pdense_conv, self).__init__()
        self.conv = nn.Sequential(
            # nn.InstanceNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, inorm=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.inorm = nn.InstanceNorm2d(out_planes, eps=1e-5, momentum=0.1, affine=False)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Spatial_attn_layer(nn.Module):
    def __init__(self, out_ch, kernel_size=5):
        super(Spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, out_ch, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=True, inorm=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return scale

class Dorsal_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dorsal_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            # nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=False),  # nn.InstanceNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        main = self.conv(x)
        res = self.res(x)
        out = main + res
        out = self.act(out)
        return out

class Ventral_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Ventral_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Fusion_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        fusion = torch.cat((x, y), dim=1)
        out = self.conv(fusion)
        return out

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## define the basic component of RDB
class DB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kernel_size = 3):
        super(DB_Conv, self).__init__()
        n_feats = inChannels
        rate  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(n_feats, rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU()
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

## define the dense block (DB)
class DB(nn.Module):
    def __init__(self, n_layer):
        super(DB, self).__init__()
        n_feats = 32
        rate  = 32
        kernel_size = 3
        
        convs = []
        for n in range(n_layer):
            convs.append(DB_Conv(n_feats + n * rate, rate))
        self.convs = nn.Sequential(*convs)
        
        self.LFF = nn.Conv2d(n_feats + n_layer * rate, n_feats, 1, padding=0, stride=1)
    
    def forward(self, x):
        out = self.LFF(self.convs(x)) + x
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## define the Edge-Net
class Edge_Net(nn.Module):
    def __init__(self, n_feats, n_layer, kernel_size=3, conv=default_conv):
        super(Edge_Net, self).__init__()

        self.trans = conv(3, n_feats, kernel_size)
        self.head = ResBlock(conv, n_feats, kernel_size)
        self.rdb = DB(n_layer)
        self.tail = ResBlock(conv, n_feats, kernel_size)
        self.rebuilt = conv(n_feats, 3, kernel_size)

    def forward(self, x):
        out = self.trans(x)
        # out = self.head(out)
        out = self.rdb(out)
        # out = self.tail(out)
        out = self.rebuilt(out)
        out = x + out
        return out

#########################################
class BioInspiredLittleAddInhibitionBlock(nn.Module):
    def __init__(self, in_channel, inner_channel, out_channel, strides=1):
        super(BioInspiredLittleAddInhibitionBlock, self).__init__()
        self.strides = strides
        self.MidgetCells = nn.Sequential(
            nn.Conv2d(in_channel, inner_channel, kernel_size=3, stride=strides, padding=1),  # kernel_size=5 or 7
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, out_channel, kernel_size=1, stride=strides),
            nn.ReLU(inplace=True)
        )
        self.ParasolCells = nn.Sequential(
            nn.Conv2d(in_channel, inner_channel, kernel_size=1, stride=strides, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, out_channel, kernel_size=1, stride=strides),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        midget = self.MidgetCells(x)
        parasol = self.ParasolCells(x)
        out = midget + parasol
        return out

class MLGN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MLGN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class PLGN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PLGN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DPN(nn.Module):
    def __init__(self):
        super(DPN, self).__init__()
        self.mdense1 = Mdense_conv(3, 32)
        self.pdense1 = Pdense_conv(3, 32)
        self.mdense2 = Mdense_conv(32, 3)
        self.pdense2 = Pdense_conv(32, 3)
        self.mlgn = MLGN(32, 32)
        self.plgn = PLGN(32, 32)
        self.mv1 = DeformConv2d(3, 16, kernel_size=5, padding=2)
        self.mv2 = DeformConv2d(16, 32, kernel_size=5, padding=2)
        self.mv3 = DeformConv2d(32, 3, kernel_size=5, padding=2)
        # self.mv4 = DeformConv2d(16, 3, kernel_size=3, padding=1)
        self.pv1 = DeformConv2d(3, 16, kernel_size=5, padding=2)
        self.pv2 = DeformConv2d(16, 32, kernel_size=5, padding=2)
        self.pv3 = DeformConv2d(32, 3, kernel_size=5, padding=2)
        # self.pv4 = DeformConv2d(16, 3, kernel_size=3, padding=1)

        self.Edge_Net1 = Edge_Net(32, 3)
        self.Edge_Net2 = Edge_Net(32, 3)

        self.sa1 = Spatial_attn_layer(out_ch=3)
        self.sa2 = Spatial_attn_layer(out_ch=16)
        self.sa3 = Spatial_attn_layer(out_ch=32)
        self.sa4 = Spatial_attn_layer(out_ch=16)
        self.sa5 = Spatial_attn_layer(out_ch=8)
        self.conv1 = nn.Conv2d(3, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(3, 16, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, 1, padding=0)
        self.conv4 = nn.Conv2d(32, 16, 1, padding=0)
        self.conv5 = nn.Conv2d(16, 8, 1, padding=0)
        # self.conv4 = nn.Conv2d(3, 32, 1, padding=0)
        # self.conv5 = nn.Conv2d(6, 16, 1, padding=0)
        # self.conv6 = nn.Conv2d(16, 32, 1, padding=0)
        self.ventral_conv1 = Ventral_conv(6, 3)
        self.ventral_conv2 = Ventral_conv(6, 16)
        self.ventral_conv3 = Ventral_conv(16, 32)
        self.ventral_conv4 = Ventral_conv(32, 16)
        self.ventral_conv5 = Ventral_conv(16, 8)
        self.dorsal_conv1 = Dorsal_conv(3, 16)
        self.dorsal_conv2 = Dorsal_conv(32, 32)
        self.fusion = Fusion_conv(64, 16)

    
    def forward(self, x):
        mDenseOut1 = self.mdense1(x)
        pDenseOut1 = self.pdense1(x)
        mDenseOut = self.mdense2(mDenseOut1) + x
        pDenseOut = self.pdense2(pDenseOut1) + x
        
        medge = self.Edge_Net1(mDenseOut) + mDenseOut
        pedge = self.Edge_Net1(pDenseOut) + pDenseOut

        mDenseOut = self.mv3(self.mv2(self.mv1(mDenseOut))) + medge + mDenseOut
        pDenseOut = self.mv3(self.mv2(self.mv1(pDenseOut))) + pedge + pDenseOut

        ventralInput = torch.cat((mDenseOut, pDenseOut), dim=1)
        # ventralInput = pDenseOut
        sa1 = self.sa1(mDenseOut) + self.conv1(mDenseOut)
        ventralconv1 = self.ventral_conv1(ventralInput)
        ventralconv1 = torch.cat((ventralconv1, sa1), dim=1)

        sa2 = self.sa2(sa1) + self.conv2(sa1)
        # dorsalconv1 = self.dorsal_conv1(mDenseOut)
        ventralconv2 = self.ventral_conv2(ventralconv1)

        dorsalconv2 = torch.cat((ventralconv2, sa2), dim=1)
        sa3 = self.sa3(dorsalconv2) + self.conv3(dorsalconv2)
        # dorsalconv3 = self.dorsal_conv2(dorsalconv2)
        ventralconv3 = self.ventral_conv3(ventralconv2)

        out = self.fusion(sa3, ventralconv3)

        return out