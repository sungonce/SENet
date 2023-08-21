# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np
import test.test_loader as loader
from test.evaluate import compute_map

@torch.no_grad()
def extract_feature(model, data_dir, dataset, gnd_fn, split, scale_list):
    with torch.no_grad():
        test_loader = loader.construct_loader(data_dir, dataset, gnd_fn, split, scale_list)
        img_feats = [[] for i in range(len(scale_list))] 

        for im_list in tqdm(test_loader):
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                desc = model.extract_global_descriptor(im_list[idx])
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                desc = F.normalize(desc, p=2, dim=1)
                img_feats[idx].append(desc.detach().cpu())

        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
        img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg

@torch.no_grad()
def test_revisitop(cfg, ks, ranks):
    # revisited evaluation
    gnd = cfg['gnd']
    ranks_E, ranks_M, ranks_H = ranks

    # evaluate ranks
    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks_E, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks_M, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks_H, gnd_t, ks)

    return (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH)