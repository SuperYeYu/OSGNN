import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp

def load_ACM_data(prefix=r'.\Dataset\ACM'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    PAP = scipy.sparse.load_npz(prefix + '/pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    PAP = F.normalize(PAP, dim=1, p=2)

    PSP = scipy.sparse.load_npz(prefix + '/psp.npz').A
    PSP = torch.from_numpy(PSP).type(torch.FloatTensor)
    PSP = F.normalize(PSP, dim=1, p=2)

    G = [PAP, PSP]

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    labels = torch.LongTensor(labels)
    type_mask = np.load(prefix + '/node_types.npy')

    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask




def load_IMDB_data(prefix=r'.\Dataset\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    MAM = np.load(prefix + '/mam.npy')
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    MAM = F.normalize(MAM, dim=1, p=2)

    MDM = np.load(prefix + '/mdm.npy')
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    MDM = F.normalize(MDM, dim=1, p=2)

    G = [MAM, MDM]

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    labels = torch.LongTensor(labels)
    type_mask = np.load(prefix + '/node_types.npy')

    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask



def load_DBLP_data(prefix=r'.\Dataset\DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20)

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    APA = scipy.sparse.load_npz(prefix + '/apa.npz').A
    APA = torch.from_numpy(APA).type(torch.FloatTensor)
    APA = F.normalize(APA, dim=1, p=2)

    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').A
    APTPA = torch.from_numpy(APTPA).type(torch.FloatTensor)
    APTPA = F.normalize(APTPA, dim=1, p=2)

    APCPA = scipy.sparse.load_npz(prefix + '/apvpa.npz').A
    APCPA = torch.from_numpy(APCPA).type(torch.FloatTensor)
    APCPA = F.normalize(APCPA, dim=1, p=2)

    G = [APA, APTPA, APCPA]

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features = [features_0, features_1, features_2, features_3]

    labels = torch.LongTensor(labels)
    type_mask = np.load(prefix + '/node_types.npy')

    num_classes = 4
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask

