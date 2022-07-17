import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
from models import *
from torch.nn.parameter import Parameter
# import spectral
import modelStatsRecord
# from torchsummary import summary
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 128)
parser.add_argument("-c","--src_input_dim",type = int, default = 100)
parser.add_argument("-d","--tar_input_dim",type = int, default = 100) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 9)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
# target
parser.add_argument("-m","--test_class_num",type=int, default=9)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets', 'TRIAN_META_DATA_imdb_ocbs.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
source_imdb['data']=np.array(source_imdb['data'])
source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
source_imdb['set']=np.array(source_imdb['set'],dtype='int')
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']# (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(np.array(source_imdb['data']).shape) # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
last_dir = os.path.abspath(os.path.dirname(os.getcwd()))
test_data = last_dir+'/HSI_META_DATA/test_ocbs/paviau_data.mat'
test_label = last_dir+'/HSI_META_DATA/test_ocbs/paviau_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

# model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
def conv3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class Octave_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(Octave_3d_CNN, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel1)
        self.conv2 = conv3x3x3(out_channel1, out_channel1)
        self.conv3 = conv3x3x3(out_channel1, out_channel2)
        self.conv4 = conv3x3(528, FEATURE_DIM)

        self.final_feat_dim = FEATURE_DIM

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)

        X_H = F.relu(self.conv1(x),inplace=True)
        X_L_TMP = F.avg_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
        X_L = F.relu(self.conv1(X_L_TMP), inplace=True)
        X_H = F.avg_pool3d(X_H, kernel_size=(3,1,1), stride=(3,1,1))
        X_L = F.avg_pool3d(X_L, kernel_size=(3,1,1), stride=(3,1,1))

        X_H_H = F.relu(self.conv2(X_H),inplace=True)
        X_H_L_TMP = F.avg_pool3d(X_H, kernel_size=(1,2,2), stride=(1,2,2))
        X_H_L = F.relu(self.conv2(X_H_L_TMP),inplace=True)

        X_L_H_TMP = F.interpolate(X_L,size=(X_L.shape[2],2*X_L.shape[3]+1,2*X_L.shape[4]+1),mode='trilinear',align_corners=False)
        X_L_H = F.relu(self.conv2(X_L_H_TMP),inplace=True)
        X__L_L = F.relu(self.conv2(X_L),inplace=True)
        Y_H = torch.add(X_H_H,X_L_H)
        Y_L = torch.add(X_H_L,X__L_L)

        Y_H_L_TMP = F.avg_pool3d(Y_H,kernel_size=(1,2,2), stride=(1,2,2))
        Y_H_L = F.relu(self.conv3(Y_H_L_TMP),inplace=True)
        Y_L_L = F.relu(self.conv3(Y_L),inplace=True)
        Y = torch.add(Y_H_L,Y_L_L)

        Z_TMP = Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2],Y.shape[3],Y.shape[4])
        Z = F.relu(self.conv4(Z_TMP), inplace=True)
        flatten = F.adaptive_avg_pool2d(Z,(1,1))

        embedding_feature = flatten.view(flatten.shape[0],-1) #(1,160)

        return embedding_feature
#SPRM
class SRMLayer(nn.Module):

    def __init__(self, channel):
        super(SRMLayer, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(5,3), stride=(1,3),padding=(2,1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, eps=1e-5):
        # feature descriptor on the global spatial information
        N, C, _, _ = x.size()
        channel_center = x.view(N, C, -1)[:, :, int((x.shape[2] * x.shape[3] - 1) / 2)]
        channel_center = channel_center.unsqueeze(2)
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std, channel_center), dim=2)

        y = self.conv(t.unsqueeze(1)).transpose(1, 2)
        y = self.sigmoid(y)

        return x * (1+y.expand_as(x))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.feature_encoder = Octave_3d_CNN(1,8,16)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.target_mapping = SRMLayer(N_DIMENSION)
        self.source_mapping = SRMLayer(N_DIMENSION)

    def forward(self, x, domain='source'):  # x
        # print(x.shape)
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        # print(x.shape)#torch.Size([45, 100, 9, 9])
        feature = self.feature_encoder(x)  # (45, 64)
        # print((feature.shape))
        return feature, feature

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
latest_G,latest_RandPerm,latest_Row, latest_Column,latest_nTrain = None,None,None,None,None
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    print(get_parameter_number(feature_encoder))

    feature_encoder.apply(weights_init)

    feature_encoder.cuda()
    # summary(feature_encoder, (100, 9, 9))
    feature_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(EPISODE):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()

        # source domain few-shot
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5, 1, 15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_features, _ = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_features, _ = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            logits = MD_distance(support_features,support_labels,query_features)


            f_loss = crossEntropy(logits, query_labels.cuda())

            loss = f_loss

            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_features, _ = feature_encoder(supports.cuda(),  domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features, _ = feature_encoder(querys.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # fsl_loss
            logits = MD_distance(support_features, support_labels, query_features)
            f_loss = crossEntropy(logits, query_labels.cuda())

            loss = f_loss
            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            elapsed_time = time.time()-train_start
            train_loss.append(loss.item())
            print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}, elapsed time: {:6.4f}'.format(episode + 1, \
                                                                                                                loss.item(),
                                                                                                                querys.shape[0],
                                                                                                                total_hit / total_num,
                                                                                                                   elapsed_time))
        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            test_features_all = []
            test_labels_all = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_features, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            flag=1
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)

                if flag==1:
                    predict_logits,class_representations,class_precision_matrices = MD_distance_test1(train_features, train_labels, test_features)
                else:
                    predict_logits = MD_distance_test2(test_features,class_representations,class_precision_matrices)

                test_features_tmp = test_features.cpu().detach().numpy()
                test_features_all.append(test_features_tmp)

                predict_labels = torch.argmax(predict_logits, dim=1).cpu()
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]
                test_labels_all = np.append(test_labels_all, test_labels)

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)
                flag = flag + 1
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str( "checkpoints/DFSL_feature_encoder_" + "UP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = total_rewards / len(test_loader.dataset)
                OA = acc[iDataSet]
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
                P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end

    latest_G, latest_RandPerm, latest_Row, latest_Column, latest_nTrain = G, RandPerm, Row, Column, nTrain
    for i in range(len(predict)):  # predict ndarray <class 'tuple'>: (9729,)
        latest_G[latest_Row[latest_RandPerm[latest_nTrain + i]]][latest_Column[latest_RandPerm[latest_nTrain + i]]] = \
            predict[i] + 1
    sio.savemat('classificationMap/UP/CMFSL_UP_pred_map_latest' + '_' + repr(int(OA * 10000)) + '.mat', {'latest_G': latest_G})
    test_features_all = np.array(test_features_all)
    test_features_all = np.vstack(test_features_all)
    sio.savemat('classificationMap/UP/test_features_all' + '_' + repr(int(OA * 10000)) + '.mat',{'test_features_all': test_features_all})
    sio.savemat('classificationMap/UP/test_labels_all' + '_' + repr(int(OA * 10000)) + '.mat',{'test_labels_all': test_labels_all})
    hsi_pic_latest = np.zeros((latest_G.shape[0], latest_G.shape[1], 3))
    for i in range(latest_G.shape[0]):
        for j in range(latest_G.shape[1]):
            if latest_G[i][j] == 0:
                hsi_pic_latest[i, j, :] = [0, 0, 0]
            if latest_G[i][j] == 1:
                hsi_pic_latest[i, j, :] = [216, 191, 216]
            if latest_G[i][j] == 2:
                hsi_pic_latest[i, j, :] = [0, 255, 0]
            if latest_G[i][j] == 3:
                hsi_pic_latest[i, j, :] = [0, 255, 255]
            if latest_G[i][j] == 4:
                hsi_pic_latest[i, j, :] = [45, 138, 86]
            if latest_G[i][j] == 5:
                hsi_pic_latest[i, j, :] = [255, 0, 255]
            if latest_G[i][j] == 6:
                hsi_pic_latest[i, j, :] = [255, 165, 0]
            if latest_G[i][j] == 7:
                hsi_pic_latest[i, j, :] = [159, 31, 239]
            if latest_G[i][j] == 8:
                hsi_pic_latest[i, j, :] = [255, 0, 0]
            if latest_G[i][j] == 9:
                hsi_pic_latest[i, j, :] = [255, 255, 0]
    utils.classification_map(hsi_pic_latest[4:-4, 4:-4, :] / 255, latest_G[4:-4, 4:-4], 24,
                             'classificationMap/UP/CMFSL_UP_pred_map_latest'+ '_' + repr(int(OA * 10000))+'.png')

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
###
ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
TRAINING_TIME_RES_SS4 = np.transpose(training_time)
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM
ITER = nDataSet

modelStatsRecord.outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                              ELEMENT_PRE_RES_SS4, AP_RES_SS4,
                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                              classes_num, ITER,
                              './records/pavia_result_train_iter_times_{}shot.txt'.format(TEST_LSAMPLE_NUM_PER_CLASS))


