from common import *
import matplotlib.pyplot as plt
from network2tikz import plot

def main():
    tests = [
        ('liv', 'eng', './lang-liv/src/fst/stems/'),
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

        G_temp = nx.Graph()
        word = 'liv_Japān'
        G_temp.add_node(word)
        all_edges = []
        prediction_edges = []
        to_predict = []
        for n in G.neighbors(word):
            G_temp.add_node(n)
            G_temp.add_edge(word, n)
            all_edges.append((word, n))

            for nn in G.neighbors(n):
                G_temp.add_node(nn)
                G_temp.add_edge(n, nn)
                all_edges.append((n, nn))
                if nn.startswith('eng_'):
                    to_predict.append((word, nn))

        preds = jaccard_coefficient(G_temp, to_predict)
        preds = list(set(preds))
        preds = [p for p in preds if p[2] > 0]
        for u, v, p in preds:
            prediction_edges.append((u, v))

        # pos = nx.random_layout(G_temp)
        val_map = {'rus_': 1.0,
                   'liv_': 0.5714285714285714,
                   'H': 0.0}
        values = [val_map.get(node[:4], 0.25) for node in G_temp.nodes()]
        plt.figure(figsize=(14, 5))
        ax = plt.gca()
        ax.axis('off')

        pos = nx.spring_layout(G_temp) #, k=0.05, iterations=100)

        nx.draw_networkx_nodes(G_temp, pos, ax=ax, node_size=[len(v) * 550 for v in G_temp.nodes()], node_color='w', alpha=0.75,
                               edgecolors='black')
        nx.draw_networkx_labels(G_temp, pos)
        nx.draw_networkx_edges(G_temp, pos, ax=ax, edgelist=prediction_edges, style='dashed', edge_color='r', arrows=False)
        nx.draw_networkx_edges(G_temp, pos, ax=ax, edgelist=all_edges, arrows=False)
        # plot(G_temp,'graph.tex')

        l, r = plt.xlim()
        # plt.xlim(l - 2, r + 2)
        plt.tight_layout()
        plt.savefig("graph.pdf", bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == '__main__':
    main()
