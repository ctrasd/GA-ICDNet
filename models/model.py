import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchvision.transforms import *
from tensorboardX import SummaryWriter
import torchvision.models as models
import torchvision
from torch.nn.parameter import Parameter
from torch.nn import functional as F, ParameterList, ModuleList, Parameter

class Part_feature(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super(Part_feature, self).__init__()
        self.feature_dim = feature_dim
        self.pool=nn.MaxPool2d(2,2)
        self.conv=nn.Conv2d(feature_dim, out_dim, 4, 1, 0)
        
    def forward(self, x):
        x=self.pool(x)
        x=self.conv(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class p_norm(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(p_norm,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        x=x.clamp(min=eps).pow(p)
        y=x.sum(dim=1,keepdim=True)
        x=x/y
        return x
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0])

class Inceptionv1(nn.Module):
    def __init__(self, in_channels,out_channels1, out_channels2reduce,out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(Inceptionv1, self).__init__()

        self.branch1_conv = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels2reduce,kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce,out_channels=out_channels2,kernel_size=3)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

    def forward(self,x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def ConvBNReLUFactorization(in_channels,out_channels,kernel_sizes,paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV2ModuleB(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1,3],paddings=[0,1]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3,1],paddings=[1, 0]),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3,kernel_sizes=[1, 3], paddings=[0, 1]),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV2ModuleC(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
        
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1,3],paddings=[0,1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3,1],paddings=[1,0])

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3,stride=1,padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1],paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3],paddings=[0, 1])

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        downsample = nn.Sequential(
                conv1x1(inplanes, planes *self. expansion, stride),
                nn.BatchNorm2d(planes *self. expansion),
            )
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ICDNet_group_mask_mask_early_8(nn.Module):
    def __init__(self, alpha=0.1):
        super(ICDNet_group_mask_mask_early_8, self).__init__()
        self.relu = nn.ReLU()
        #encoder
        self.conv1 = nn.Conv2d(1, 32, 3, 1,1)
################################*****************
        
        #res
        self.inception1 = Bottleneck(32, 64)
        self.maxpooling1 = nn.MaxPool2d(2,2)
        self.inception2 = Bottleneck(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2,2)
        #self.inception3_ident = Bottleneck(96,192)
        #self.inception3_bag = Bottleneck(32,64)
        self.inception3= Bottleneck(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2, 2)
        self.inception4_ident = Bottleneck(192,384)
        self.inception4_bag = Bottleneck(64,128)
        self.maxpooling4 = nn.MaxPool2d(2, 2) 
        '''
        #b4
        self.inception1 = InceptionV2ModuleB(32, 16, 32, 32, 16, 8, 8)
        self.maxpooling1 = nn.MaxPool2d(2,2)
        self.inception2 = InceptionV2ModuleB(64, 32, 64, 64, 32, 16, 16)
        self.maxpooling2 = nn.MaxPool2d(2,2)
        self.inception3 = InceptionV2ModuleB(128, 64, 128, 128, 64, 32, 32)
        
        self.inception4_bag = InceptionV2ModuleB(64, 48, 64, 52, 36 , 12, 16)
        self.inception4_ident = InceptionV2ModuleB(192, 144, 192, 156, 108 , 36, 48)
        self.maxpooling4 = nn.MaxPool2d(2, 2)
        self.maxpooling3 = nn.MaxPool2d(2, 2)

        '''

        self.conv5_ident = nn.Conv2d(384, 96, 4, 1, 0)
        self.conv5_bag_cla = nn.Conv2d(128, 32, 4, 1, 0)

        self.group_conv=[]
        for i in range(4):
            self.group_conv.append(Part_feature(128,8))
        self.group_conv = ModuleList(self.group_conv)

        self.deconv1 = nn.ConvTranspose2d(128, 512, 4, 2, 0)
        self.de_bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.de_bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.de_bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.de_bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.tanh5 = nn.Tanh()

        self.classifiers = nn.Linear(in_features=32, out_features=7)

        self.classifiers_4channel=nn.Linear(in_features=7, out_features=4,bias=True)

        self.sa=SpatialAttention(3)
        self.sa2=SpatialAttention(3)
        self.gamma_ident=nn.Parameter(torch.ones(1))
        self.gamma_bag=nn.Parameter(torch.ones(1))
        self.vconv_ident=nn.Conv2d(in_channels = 96 , out_channels = 96,kernel_size= 1)
        self.vconv_bag=nn.Conv2d(in_channels = 32, out_channels = 32,kernel_size= 1)

    def forward(self, x, isTest = False):
        # encoder
        x = self.conv1(x) #32
        x = self.relu(x) 
        x = self.inception1(x) #64
        x = self.maxpooling1(x) #64*32*32
        x = self.inception2(x) #128*32*32
        x = self.maxpooling2(x)#128*16*16
        x = self.inception3(x) #256*16*16
        x = self.maxpooling3(x) #256*8*8
        x_se_ident=x[:,:192,:].clone()
        x_se_bag=x[:,192:,:].clone()
        #x_se_bag,att_bag=self.sa(x_se_bag)
        mask_8=self.sa(x_se_bag)
        #mask_8_id=self.sa2(x_se_ident)
        x_se_bag=x_se_bag+self.gamma_bag*x_se_bag*mask_8
        x_se_ident=x_se_ident-self.gamma_ident*x_se_ident*mask_8
        


        
        x_se_bag = self.inception4_bag(x_se_bag) #128*8*8
        x_bag_feature=x_se_bag.clone()
        x_se_ident = self.inception4_ident(x_se_ident) #128*8*8
        x_se_bag=self.maxpooling4(x_se_bag)
        
        x_bag=self.conv5_bag_cla(x_se_bag) #batch*32*1*1
        
        
        x_se_ident = self.maxpooling4(x_se_ident) #384*4*4
        x_ident = self.conv5_ident(x_se_ident) # batch*96*1*1
        
        features=[]
        for i in range(4):
            features.append(self.group_conv[i](x_bag_feature))
        features_bag=torch.cat([j for j in features], dim=1) # batch*32*1*1

        x=torch.cat([x_ident,features_bag],dim=1) #batch*128*1*1

        x_bag=x_bag.squeeze(2)
        x_bag=x_bag.squeeze(2)
        x_bag=self.classifiers(x_bag) #batch*7
        x_bag_soft=x_bag.clone()
        x_bag_soft=self.classifiers_4channel(1-x_bag_soft)
        for i in range(4):
            features[i]=features[i]*(x_bag_soft[:,i].unsqueeze(1).expand(-1,8).unsqueeze(2).unsqueeze(3))
        features=torch.cat([j for j in features], dim=1) 
        x_ident=torch.cat([x_ident,features],dim=1)


        ident_features = x_ident
        x_ident = self.deconv1(x_ident)
        x_ident = self.de_bn1(x_ident)
        x_ident = self.relu(x_ident)
        x_ident = self.deconv2(x_ident)
        x_ident = self.de_bn2(x_ident)
        x_ident = self.relu(x_ident)
        x_ident = self.deconv3(x_ident)
        x_ident = self.de_bn3(x_ident)
        x_ident = self.relu(x_ident)
        x_ident = self.deconv4(x_ident)
        x_ident = self.de_bn4(x_ident)
        x_ident = self.relu(x_ident)
        x_ident = self.deconv5(x_ident)
        x_ident = self.tanh5(x_ident)

        x = self.deconv1(x)
        x = self.de_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.de_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.de_bn3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.de_bn4(x)
        x = self.relu(x)
        x = self.deconv5(x)
        x = self.tanh5(x)
        return x_ident, x, ident_features,x_bag

def icdnet_group_mask_mask_early_8():
    model = ICDNet_group_mask_mask_early_8()
    return model
