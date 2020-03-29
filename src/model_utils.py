import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
#         self.prelu1_1 = nn.PReLU()
#         self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
#         self.prelu1_2 = nn.PReLU()
#         self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.prelu2_1 = nn.PReLU()
#         self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
#         self.prelu2_2 = nn.PReLU()
#         self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
#         self.prelu3_1 = nn.PReLU()
#         self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
#         self.prelu3_2 = nn.PReLU()
#         self.preluip1 = nn.PReLU()
#         self.ip1 = nn.Linear(128 * 7 * 7, 2)
#         self.ip2 = nn.Linear(2, 2)
#
#     def forward(self, x):
#         x = self.prelu1_1(self.conv1_1(x))
#         x = self.prelu1_2(self.conv1_2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.prelu2_1(self.conv2_1(x))
#         x = self.prelu2_2(self.conv2_2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.prelu3_1(self.conv3_1(x))
#         x = self.prelu3_2(self.conv3_2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 128 * 7 * 7)
#         ip1 = self.preluip1(self.ip1(x))
#         ip2 = self.ip2(ip1)
#         print(ip2.shape)
#         return ip1, ip2

#没有分解通道的三通道网络
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.BatchNorm2d(192),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.BatchNorm2d(384),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256*6*6, 4096),
#             nn.Dropout(0.3),
#             nn.LeakyReLU(inplace=True),
#
#             nn.Linear(4096, 2),
#             nn.Dropout(0.3),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.classifier = nn.Linear(2, 2)
#
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out1 = self.conv2(out1)
#         out1 = self.conv3(out1)
#         out1 = self.conv4(out1)
#         out1 = self.conv5(out1)   #[b, 256, 6, 6]
#         #print("out:", out.shape)
#         out = torch.flatten(out1, 1)
#         x = self.fc(out)           # x是输出的特征，y是输出的标签值
#         #print(x.shape)
#         y = self.classifier(x)
#         return x, y

#加载网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*6*6, 4096),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),

            nn.Linear(4096, 2),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True)
        )
        self.classifier = nn.Linear(2, 2)


    def forward(self, x1, x2, x3):
        out1 = self.conv1(x1)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)
        out1 = self.conv4(out1)
        out1 = self.conv5(out1)   #[b, 256, 6, 6]
        #print("out1:", out1.shape)
        out2 = self.conv1(x2)
        out2 = self.conv2(out2)
        out2 = self.conv3(out2)
        out2 = self.conv4(out2)
        out2 = self.conv5(out2)   #[b, 256, 6, 6]
        out3 = self.conv1(x3)
        out3 = self.conv2(out3)
        out3 = self.conv3(out3)
        out3 = self.conv4(out3)
        out3 = self.conv5(out3)   #[b, 256, 6, 6]
        out = torch.cat([out1, out2[:, :128, :, :], out3[:, :128, :, :]], dim=1)
        #print("out:", out.shape)
        out = torch.flatten(out, 1)
        x = self.fc(out)           # x是输出的特征，y是输出的标签值
        y = self.classifier(x)
        return x, y
    
class RingLoss(nn.Module):
    """
    Refer to paper
    Ring loss: Convex Feature Normalization for Face Recognition
    """
    def __init__(self, type='L2', loss_weight=1.0):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data[0])
        if self.type == 'L1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss
    

class COCOLoss(nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, alpha=6.25):
        super(COCOLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha*nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))

        return logits


class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)# n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)


        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot)#.cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood


class LGMLoss_v0(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """
    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))


    def forward(self, feat, label):
        batch_size = feat.shape[0]

        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)
        margin_logits = -0.5 * margin_dist
        logits = -0.5 * dist

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        likelihood = (1.0/batch_size) * cdiff.pow(2).sum(1).sum(0) / 2.0
        return logits, margin_logits, likelihood


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, y, feat):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers)


class CenterlossFunction(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long()) # Eq. 3

        # init every iteration
        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()
        # print counts, grad_centers

        # Eq. 4 || need optimization !! To be vectorized, but how?
        for i in range(feature.size(0)):
            j = int(label[i].item())
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers

# net = Net()
# x = torch.randn(5, 1, 224, 224)
# out = net(x, x, x)
# print(out)
