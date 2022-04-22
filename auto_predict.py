import networkx as nx
from common import *
from collections import defaultdict
import unidecode
import os


def main():
    tests = [
        ('liv', 'eng', './lang-liv/src/fst/stems/'),
        ('kpv', 'eng', './kpv/'),
        ('kpv', 'fra', './kpv/'),
        ('myv', 'eng', None),
        # ('sms', 'eng', './lang-sms/src/fst/stems/'),
    ]

    langs = ['fra', 'rus', 'fin', 'est', 'lav']

    pred_fns = [
        jaccard_coefficient,
        resource_allocation_index,
        adamic_adar_index,
        preferential_attachment,
    ]  # common_neighbor_centrality]

    for test in tests:
        src, tgt, path = test

        to_pred = []

        # create initial graph
        if path:
            G = main_graph(path=path, ignore=(tgt,))
            main_n = G.nodes()
        else:
            G = nx.Graph()
            G, main_n = add_translation(G, src, ignore=(tgt,))

        for lang in langs:
            G, _ = add_wiki(G, lang)  # add wiktionaries

        # for all nodes with tgt language
        for node in main_n:
            if not node.startswith('{}_'.format(src)):
                continue

            for _nn in G.neighbors(node):
                if _nn.startswith('{}_'.format(tgt)):
                    to_pred.append((node, _nn))

                for __nn in G.neighbors(_nn):
                    if __nn.startswith('{}_'.format(tgt)):
                        to_pred.append((node, __nn))

        try:
            os.mkdir('./predictions/{}/'.format(src))
        except OSError as error:
            pass

        for pred_fn in pred_fns:
            predict_and_write(G, to_pred, pred_fn, './predictions/{}/{}_{}.txt'.format(src, tgt, pred_fn.__name__),
                              sample=None)
    #
    # preds = predict_and_write(G, to_pred, pred_fn, 'preds_{}_{}.txt'.format(src, tgt))
    # preds_x = defaultdict(list)
    # preds_y = dict()
    # for a in preds:
    #     if a[2] >= 0.5:
    #         preds_x[a[0]].append(a[1])
    #     preds_y[(a[0], a[1],)] = a[2]
    # G, main_n = add_translation(G, src)
    #
    # true_n = 0
    # total = 0
    #
    # for x in preds:
    #     s, t, score = x
    #     if score < 0.5:
    #         continue
    #     if not G.has_node(s):
    #         continue
    #     n = list(G.neighbors(s))
    #     if t in n:
    #         true_n += 1
    #     else:
    #         print(x, n)
    #     total += 1

    # for s, t in preds_x.items():
    #     if not G.has_node(s):
    #         continue
    #     for n in G.neighbors(s):
    #         if not n.startswith('{}_'.format(tgt)):
    #             continue
    #         if n in t:
    #             true_n += 1
    #             print(s, n, t, preds_y[(s, n)])
    #         total += 1
    # for n in main_n:
    #     if not n.startswith('{}_'.format(src)) or not preds_x[n]:
    #         continue
    #     for nn in G.neighbors(n):
    #         if not nn.startswith('{}_'.format(tgt)):
    #             continue
    #         if nn in preds_x[n]:
    #             true_n += 1
    #         else:
    #             print(n, nn, preds_x[n])
    #         total += 1
    # print(true_n / total, total)


if __name__ == '__main__':
    main()
