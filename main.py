import argparse
import torch
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from data import load_ACM_data, load_IMDB_data, load_DBLP_data
import numpy as np
import random
from model_han_gl import OSGNN_GL


def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args):
    if args['dataset'] == "ACM":
        G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask = load_ACM_data()
        norm = [1, 1]
        if args['HSIC'] == True:
            lam = 1e-16
        else:
            lam = 0
    elif args['dataset'] == "DBLP":
        G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask = load_DBLP_data()
        norm = [1, 0]
        if args['HSIC'] == True:
            lam = 1e-13
        else:
            lam = 0
    elif args['dataset'] == "IMDB":
        G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask = load_IMDB_data()
        norm = [2, 0]
        if args['HSIC'] == True:
            lam = 1e-13
        else:
            lam = 0

    in_dims = [features.shape[1] for features in features]
    features = [feat.to(args['device']) for feat in features]
    labels = labels.to(args['device'])
    G = [g.to(args['device']) for g in G]
    ADJ = ADJ.to(args['device'])
    model = OSGNN_GL(in_dims, args['hidden_units'], len(G), args['dropout'], num_classes).to(args['device'])

    G = [graph.to(args['device']) for graph in G]

    early_stopping = EarlyStopping(patience=args['patience'], verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):

        model.train()
        logits, h, loss_dependence = model(features, G, ADJ, type_mask, norm, args['agg'])
        loss = loss_fcn(logits[train_idx], labels[train_idx]) + lam*loss_dependence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits, h, loss_dep = model(features, G, ADJ, type_mask, norm, args['agg'])
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])
        print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),val_loss.item(),test_loss.item()))
        early_stopping(val_loss.data.item(), model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    print('\ntesting...')
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
    model.eval()
    logits, h, _ = model(features, G, ADJ, type_mask, norm, args['agg'])
    evaluate_results_nc(logits[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OSGNN')
    parser.add_argument('--dataset', default='DBLP')
    parser.add_argument('--agg', default='contact')
    parser.add_argument('--HSIC', default=False)
    parser.add_argument('--lr', default=0.002)
    parser.add_argument('--num_heads', default=[8])
    parser.add_argument('--hidden_units', default=64)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--num_epochs', default=1000)
    parser.add_argument('--weight_decay', default=0.00)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args().__dict__
    set_random_seed()
    print(args)
    main(args)

