# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import numpy as np

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop

@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list):
    torch.backends.cudnn.benchmark = False
    model.eval()
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with SENet,'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset
        print("extract query features")
        Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", scale_list)
        print("extract database features")
        X = extract_feature(model, data_dir, dataset, gnd_fn, "db", scale_list)

        cfg = config_gnd(dataset,data_dir)

        # perform search
        print("perform global retrieval")
        sim = np.dot(X, Q.T)
        ranks = np.argsort(-sim, axis=0)

        # revisited evaluation
        ks = [1, 5, 10]
        (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=1), np.around(mapM*100, decimals=1), np.around(mapH*100, decimals=1)))
 
    torch.backends.cudnn.benchmark = True

