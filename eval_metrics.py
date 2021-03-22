# -*- coding: utf-8 -*
from __future__ import print_function, absolute_import
import numpy as np
import copy
import faiss
from tqdm import tqdm
import random
def evaluate_rank(gf,qf,g_pids,q_pids,max_rank=50):
    num_q, num_g = len(q_pids),len(g_pids)
    gf=gf.numpy()
    qf=qf.numpy()
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    index = faiss.IndexFlatL2(128)
    index.add(gf)
    print('Index build finished,Num of index:',index.ntotal)
    D, I = index.search(qf, max_rank)
    all_cmc = []
    print(I[0],g_pids[I[0][0]],q_pids[0])
    for i in range(len(I)):
        for j in range(len(I[i])):
            I[i][j]=g_pids[I[i][j]]
            I[i][j]=(I[i][j]==q_pids[i])
        I[i]=np.asarray(I[i]).astype(np.int32)
        I[i]=I[i].cumsum()
        I[i][I[i]>1]=1
        all_cmc.append(I[i])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc

def evaluate_rank_2048(gf,qf,g_pids,q_pids,max_rank=50):
    num_q, num_g = len(q_pids),len(g_pids)
    gf=gf.numpy()
    qf=qf.numpy()
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    index = faiss.IndexFlatL2(2048)
    index.add(gf)
    print('Index build finished,Num of index:',index.ntotal)
    D, I = index.search(qf, max_rank)
    all_cmc = []
    print(I[0],g_pids[I[0][0]],q_pids[0])
    for i in range(len(I)):
        for j in range(len(I[i])):
            I[i][j]=g_pids[I[i][j]]
            I[i][j]=(I[i][j]==q_pids[i])
        I[i]=np.asarray(I[i]).astype(np.int32)
        I[i]=I[i].cumsum()
        I[i][I[i]>1]=1
        all_cmc.append(I[i])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc


def evaluate(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    print("computing")
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # compute cmc curve
        orig_cmc = matches[q_idx][:] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc[0:max_rank].cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        # num_rel = orig_cmc.sum()
        # tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        # AP = tmp_cmc.sum() / num_rel
        # AP = 1.0 / (np.where(orig_cmc > 0)[0][0] + 1)
        # all_AP.append(AP)
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP