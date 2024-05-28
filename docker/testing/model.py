import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision
import torch.nn as nn
import torch
from dependency import *
from utils import get_parameter_number
import torch.nn.functional as F
import torchvision.ops.focal_loss
# from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

sigmoid = nn.Sigmoid()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss

# class BCEFocalLoss(torch.nn.Module):
#     def __init__(self, alpha=None, gamma=2):
#         super(BCEFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
#         return loss

class MultiCEFocalLoss(torch.nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, weight=None):
        super(MultiCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class 7PGCN(nn.Module):
    def __init__(self, class_list):
        ## model file for test/inference stage only
        super(7PGCN, self).__init__()
        self.num_label = class_list[0]
        self.num_pn = 1
        self.num_str = 1
        self.num_pig = 1
        self.num_rs = 1
        self.num_dag = 1
        self.num_bwv = 1
        self.num_vs = 1
        self.dropout = nn.Dropout(0.3)

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)

        ## created in graph network
        # ini_range_values = torch.rand(1, 7) * 1.5 + 0.5
        # ini_range_values = ini_range_values.view(1, -1)
        
        # if diagnosis_fusion.weight.is_cuda:
        #     ini_range_values = ini_range_values.cuda()
        # # Initialize diagnosis_fusion weights with specific values
        # with torch.no_grad():
        #     diagnosis_fusion.weight.copy_(ini_range_values)
        #     diagnosis_derm.weight.copy_(ini_range_values)
        #     diagnosis_clic.weight.copy_(ini_range_values)
        
        # define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool

        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool

        backbone_size = 2048

        self.fc_fusion_ = nn.Sequential(
            nn.Linear(backbone_size, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.derm_mlp = nn.Sequential(
            nn.Linear(backbone_size, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.clin_mlp = nn.Sequential(
            nn.Linear(backbone_size, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.fc_pn_cli = nn.Linear(128, self.num_pn)
        self.fc_str_cli = nn.Linear(128, self.num_str)
        self.fc_pig_cli = nn.Linear(128, self.num_pig)
        self.fc_rs_cli = nn.Linear(128, self.num_rs)
        self.fc_dag_cli = nn.Linear(128, self.num_dag)
        self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
        self.fc_vs_cli = nn.Linear(128, self.num_vs)

        self.fc_pn_derm = nn.Linear(128, self.num_pn)
        self.fc_str_derm = nn.Linear(128, self.num_str)
        self.fc_pig_derm = nn.Linear(128, self.num_pig)
        self.fc_rs_derm = nn.Linear(128, self.num_rs)
        self.fc_dag_derm = nn.Linear(128, self.num_dag)
        self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
        self.fc_vs_derm = nn.Linear(128, self.num_vs)

        self.fc_pn_fusion = nn.Linear(128, self.num_pn)
        self.fc_str_fusion = nn.Linear(128, self.num_str)
        self.fc_pig_fusion = nn.Linear(128, self.num_pig)
        self.fc_rs_fusion = nn.Linear(128, self.num_rs)
        self.fc_dag_fusion = nn.Linear(128, self.num_dag)
        self.fc_bwv_fusion = nn.Linear(128, self.num_bwv)
        self.fc_vs_fusion = nn.Linear(128, self.num_vs)

    def forward(self, x):
        (x_clic, x_derm) = x

        x_clic = self.conv1_cli(x_clic)
        x_clic = self.bn1_cli(x_clic)
        x_clic = self.relu_cli(x_clic)
        x_clic = self.maxpool_cli(x_clic)
        x_clic = self.layer1_cli(x_clic)
        x_clic = self.layer2_cli(x_clic)
        x_clic = self.layer3_cli(x_clic)
        x_clic = self.layer4_cli(x_clic)
        x_clic = self.avgpool_cli(x_clic)
        x_clic = x_clic.view(x_clic.size(0), -1)

        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)
        x_derm = self.layer1_derm(x_derm)
        x_derm = self.layer2_derm(x_derm)
        x_derm = self.layer3_derm(x_derm)
        x_derm = self.layer4_derm(x_derm)
        x_derm = self.avgpool_derm(x_derm)
        x_derm = x_derm.view(x_derm.size(0), -1)

        x_fusion = torch.add(x_clic, x_derm)
        x_fusion = self.fc_fusion_(x_fusion)
        x_clic = self.clin_mlp(x_clic)
        x_clic = self.dropout(x_clic)
        logit_pn_clic = self.fc_pn_cli(x_clic)
        logit_str_clic = self.fc_str_cli(x_clic)
        logit_pig_clic = self.fc_pig_cli(x_clic)
        logit_rs_clic = self.fc_rs_cli(x_clic)
        logit_dag_clic = self.fc_dag_cli(x_clic)
        logit_bwv_clic = self.fc_bwv_cli(x_clic)
        logit_vs_clic = self.fc_vs_cli(x_clic)

        x_derm = self.derm_mlp(x_derm)
        x_derm = self.dropout(x_derm)
        logit_pn_derm = self.fc_pn_derm(x_derm)
        logit_str_derm = self.fc_str_derm(x_derm)
        logit_pig_derm = self.fc_pig_derm(x_derm)
        logit_rs_derm = self.fc_rs_derm(x_derm)
        logit_dag_derm = self.fc_dag_derm(x_derm)
        logit_bwv_derm = self.fc_bwv_derm(x_derm)
        logit_vs_derm = self.fc_vs_derm(x_derm)

        x_fusion = self.dropout(x_fusion)
        logit_pn_fusion = self.fc_pn_fusion(x_fusion)
        logit_str_fusion = self.fc_str_fusion(x_fusion)
        logit_pig_fusion = self.fc_pig_fusion(x_fusion)
        logit_rs_fusion = self.fc_rs_fusion(x_fusion)
        logit_dag_fusion = self.fc_dag_fusion(x_fusion)
        logit_bwv_fusion = self.fc_bwv_fusion(x_fusion)
        logit_vs_fusion = self.fc_vs_fusion(x_fusion)

        # Combine the fusion logits into a single tensor
        logits_combined_fusion = torch.cat([sigmoid(logit_pn_fusion), sigmoid(logit_str_fusion), sigmoid(logit_pig_fusion), sigmoid(logit_rs_fusion), sigmoid(logit_dag_fusion), sigmoid(logit_bwv_fusion), sigmoid(logit_vs_fusion)], dim=1)
        logits_combined_clic = torch.cat([sigmoid(logit_pn_clic), sigmoid(logit_str_clic), sigmoid(logit_pig_clic), sigmoid(logit_rs_clic), sigmoid(logit_dag_clic), sigmoid(logit_bwv_clic), sigmoid(logit_vs_clic)], dim=1)
        logits_combined_derm = torch.cat([sigmoid(logit_pn_derm), sigmoid(logit_str_derm), sigmoid(logit_pig_derm), sigmoid(logit_rs_derm), sigmoid(logit_dag_derm), sigmoid(logit_bwv_derm), sigmoid(logit_vs_derm)], dim=1)

        return [(logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm, logits_combined_derm),
                (logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic, logit_vs_clic, logits_combined_clic),
                (logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, logit_vs_fusion, logits_combined_fusion)]


    def criterionBCE(self, logit, truth):
        m = torch.nn.Sigmoid()
        # criterion = nn.BCELoss()
        criterion = FocalLoss()
        truth = truth.float().unsqueeze(1)
        return criterion(m(logit), truth)

    def criterionMulti(self, logit, truth):
        # loss = nn.CrossEntropyLoss()(logit, truth)
        m = torch.nn.Softmax(dim=-1)
        criterion = MultiCEFocalLoss()
        return criterion(m(logit), truth)

    def criterion1(self, logit, truth):
        loss = nn.L1Loss()(logit, truth)
        return loss

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError
