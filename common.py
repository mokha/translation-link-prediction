from corpona.xml import XML
import io
from mikatools import *
from sklearn.metrics import *
from networkx import *
import re
from collections import defaultdict

prediction_re = re.compile(r'\((\w{3})_(.*), (\w{3})_(.*)\)\s+->\s+(.*)', re.I | re.U)
annotations_re = re.compile(r'\((\w{3})_(.*), (\w{3})_(.*)\)\s+->\s+(.*)\s+#([^!]+)(.*)', re.I | re.U)

lang_map = {
    'eng': ['englanti', 'inglise', 'angļu', 'английский', 'english', 'Anglais'],
    'fin': ['suomi', 'soome', 'somuigauņu', 'финский', 'finnish', 'Finnois'],
    'rus': ['venäjä', 'krievu', 'vene', 'русский', 'russian', 'Russe'],
    'est': ['eesti', 'viro', 'igauņu', 'эстонский', 'estonian', 'Estonien'],
    'fra': ['Французский', 'ranska', 'prantsuse', 'french', 'Française', 'franču'],
    'lav': ['latvialainen', 'latvian', 'Латышский', 'läti', 'Letton', 'latvietis']
}
lang_map_rev = {}
for k, v in lang_map.items():
    for _k in v:
        lang_map_rev[_k] = k


def get_xmls(path):
    import os
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        yield XML.parse_xml(os.path.join(path, filename))


def predict_and_write(G, to_pred, pred_fn, output_file, sample=None):
    import random
    preds = pred_fn(G, to_pred)
    preds = list(set(preds))
    preds = [p for p in preds if p[2] > 0]
    if sample is not None and type(sample) is int and sample > 0:
        preds = random.sample(preds, sample)
    preds = sorted(preds, key=lambda k: k[2], reverse=True)
    with io.open(output_file, 'w', encoding='utf-8') as ff:
        for u, v, p in preds:
            ff.write("(%s, %s) -> %.4f\n" % (u, v, p))
    return preds


def add_wiki(G, lang='fin', ignore=(), skip_not_present=False):
    data = json_load('./wiktionary-code/translations_{}.json'.format(lang))
    added_nodes = []
    for k, v in data.items():
        node = "{}_{}".format(lang, k.strip())
        if skip_not_present and not G.has_node(node):
            continue

        G.add_node(node)
        added_nodes.append(node)

        for _k, _v in v.items():
            for __k, __v in _v.items():
                if __k not in lang_map_rev:
                    continue

                _k_lang = lang_map_rev[__k]
                if ignore and _k_lang in ignore:
                    continue

                for ___v in __v:
                    if lang == 'lav':
                        ___v = ___v[:-3]
                    #
                    # if ' ' in ___v:
                    #     ___v = ___v.split()[0]

                    _node = "{}_{}".format(_k_lang, ___v.strip())
                    G.add_node(_node)
                    G.add_edge(_node, node)
                    added_nodes.append(_node)

    return G, added_nodes


def add_translation(G, lang='fin', ignore=(), skip_not_present=False):
    data = json_load('./dicts/translations_{}.json'.format(lang))
    added_nodes = []
    for k, v in data.items():

        node = "{}_{}".format(lang, k.strip())
        if skip_not_present and not G.has_node(node):
            continue

        G.add_node(node)
        added_nodes.append(node)

        for _k, _v in v.items():
            for __k, __v in _v.items():
                if ignore and __k in ignore:
                    continue

                for ___v in __v:
                    # if ' ' in ___v:
                    #     ___v = ___v.split()[0]
                    _node = "{}_{}".format(__k, ___v.strip())
                    G.add_node(_node)
                    G.add_edge(_node, node)
                    added_nodes.append(_node)

    return G, added_nodes


def main_graph(path, ignore=(), ):
    G = nx.Graph()
    xmls = list(get_xmls(path=path))
    for _xml in xmls:
        lang = _xml.attributes['xml:lang']
        for e in _xml['e']:
            nodes = []

            if 'lg' not in e:
                continue

            for lg in e['lg']:
                if 'l' not in lg:
                    continue

                for l in lg['l']:
                    node_str = f"{lang}_{l.text}"
                    G.add_node(node_str, **l.attributes)
                    nodes.append(node_str)

            if 'mg' not in e:
                continue

            for mg in e['mg']:
                if 'tg' not in mg:
                    continue

                for tg in mg['tg']:
                    tg_lang = tg.attributes['xml:lang']

                    if tg_lang in ignore:
                        continue

                    if 't' not in tg:
                        continue

                    for t in tg['t']:
                        node_str = f"{tg_lang}_{t.text}"
                        G.add_node(node_str, **t.attributes)

                        for n in nodes:
                            G.add_edge(n, node_str, )
    return G


def annotated_predictions():
    predictions_dir = './predictions'
    annotations_dir = './annotations'

    langs = [('kpv', 'eng',), ('myv', 'eng',), ('liv', 'eng',), ('kpv', 'fra')]

    predictions = defaultdict(list)

    for src, tgt in langs:
        with io.open('{}/{}-{}.txt'.format(annotations_dir, src, tgt), 'r', encoding='utf-8') as f:
            for l in f:
                l = l.rstrip('\n')
                result = annotations_re.match(l)
                if not result:
                    print(src, tgt, l)
                    continue

                src_l, src_w, tgt_l, tgt_w, score, tag, rest = result.groups()
                if 'good' in tag:
                    score = 1
                elif 'accept' in tag:
                    score = 1
                elif 'small' in tag:
                    score = 1
                elif 'bad' in tag:
                    score = 0
                else:
                    print(l)
                try:
                    predictions['_'.join((src_l, src_w, tgt_l, tgt_w,))].append(float(score))
                except:
                    print(l)
    for src, tgt in langs:
        for algorithm in ['adamic_adar_index', 'jaccard_coefficient',
                          'preferential_attachment',
                          'resource_allocation_index']:
            with io.open('{}/{}/{}_{}.txt'.format(predictions_dir, src, tgt, algorithm), 'r', encoding='utf-8') as f:
                for l in f:
                    l = l.rstrip('\n')
                    result = prediction_re.match(l)
                    src_l, src_w, tgt_l, tgt_w, score = result.groups()
                    key = '_'.join((src_l, src_w, tgt_l, tgt_w,))
                    if key in predictions:
                        predictions[key].append(float(score))
    return predictions


def wiki_binary_predictions():
    from itertools import permutations

    annotations_dir = './binary-data/'
    X = defaultdict(list)
    y = defaultdict(int)

    langs = ['fra', 'rus', 'fin', 'est', 'lav']
    for src, tgt in permutations(langs, 2):
        for algorithm in ['adamic_adar_index', 'jaccard_coefficient', 'preferential_attachment',
                          'resource_allocation_index']:
            with io.open('{}/{}/{}_{}.txt'.format(annotations_dir, src, tgt, algorithm), 'r',
                         encoding='utf-8') as f:
                for l in f:
                    l = l.rstrip('\n')
                    if not l:
                        continue
                    try:
                        src_w, tgt_w, classification, score = l.split('\t')

                        key = '_'.join((src_w, tgt_w))
                        X[key].append(float(score))
                        y[key] = int(classification)
                    except:
                        print(l)
    return X, y
