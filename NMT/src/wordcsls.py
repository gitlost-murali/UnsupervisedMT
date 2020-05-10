import sys, torch
# load Faiss if available (dramatically accelerates the nearest neighbor search)

dico_method = 'csls_knn_10'
dico_max_rank = 15000
dico_max_size = 0
dico_min_size = 0
dico_threshold = 0

try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def get_candidates(emb1, emb2):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if dico_max_rank > 0 and not dico_method.startswith('invsm_beta_'):
        n_src = min(dico_max_rank, n_src)

    # nearest neighbors
    if dico_method == 'nn':
        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # contextual dissimilarity measure
    elif dico_method.startswith('csls_knn_'):

        knn = dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if dico_max_size > 0:
        all_scores = all_scores[:dico_max_size]
        all_pairs = all_pairs[:dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if dico_min_size > 0:
        diff[:dico_min_size] = 1e9

    # confidence threshold
    if dico_threshold > 0:
        mask = diff > dico_threshold
#         logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        print("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def build_dictionary(src_emb, tgt_emb,cuda):
    """
    Build a training dictionary given current embeddings / mapping.
    """
#     logger.info("Building the train dictionary ...")
    # print("Building the train dictionary ...")
    s2t = True
    t2s = True
    assert s2t or t2s

    s2t_candidates = get_candidates(src_emb, tgt_emb)
    t2s_candidates = get_candidates(tgt_emb, src_emb)
    t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
    t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])

    # final_pairs = s2t_candidates | t2s_candidates # We want pairings to be separate.
    # dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))
    s2t_dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in s2t_candidates]))
    t2s_dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in t2s_candidates]))

#     logger.info('New train dictionary of %i pairs.' % dico.size(0))
    # print('New s2t train dictionary of %i pairs.' % s2t_dico.size(0))
    # print('New t2s train dictionary of %i pairs.' % t2s_dico.size(0))
    s2t_dico = s2t_dico.cuda() if cuda else s2t_dico
    t2s_dico = t2s_dico.cuda() if cuda else t2s_dico
    return (s2t_dico, t2s_dico)
