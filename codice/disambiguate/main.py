from codice.disambiguate import utils
from codice.disambiguate import graph
import nltk.tag.mapping as mapping
import json
import os
import tqdm
from nltk.stem import *
import pickle

# BABEL_KEY = '2e9efb5c-9eb4-4946-8854-cc93303168af'
BABEL_KEY = '32796f83-09b8-4c0f-8190-f57069a8f3cf'


def generate_dataset_json(files, out_file, lemma_know=None, relationship_know=None):
    d = {}
    lemmas = set()
    stemmer = PorterStemmer()

    if os.path.isfile(out_file):
        with open(out_file, 'r') as f:
            data = f.read()
            d = json.loads(data)

    if lemma_know is not None:
        d.update(lemma_know)

    if relationship_know is None:
        relationship_know = {}

    for f in files:
        with open(f) as file:
            for line in tqdm.tqdm(file):
                l = line.strip()

                tokens = l.split()
                if len(l) == 0 or tokens[12] != 'Y':
                    continue

                lemma = tokens[2]
                pos = tokens[4]

                if lemma == '%':
                    k = '%25_' + mapping.map_tag('wsj', 'universal', pos)
                else:
                    k = lemma.lower() + '_' + mapping.map_tag('wsj', 'universal', pos)
                lemmas.add(k)

    lemmas = sorted(list(lemmas))
    for l in tqdm.tqdm(lemmas):
        if l not in d:

            lemma, pos = l.rsplit('_', 1)
            print(lemma, pos, 'to add.')

            # print(mapping.map_tag('wsj', 'universal', pos))
            b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY)

            if b == -1:
                return d

            if len(b) == 0:
                b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, pos='POS')
            if len(b) == 0:
                b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, wn=False)
            if len(b) == 0:
                b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, wn=False, pos='POS')

            if len(b) == 0:
                lemma = stemmer.stem(lemma)
                print('Stemmer used: ', lemma)

                b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY)
                if len(b) == 0:
                    b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, pos='POS')
                if len(b) == 0:
                    b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, wn=False)
                if len(b) == 0:
                    b = utils.getAssociatedSynsetsBabelnet(lemma, pos, BABEL_KEY, wn=False, pos='POS')

            print(b)
            # assert(len(b) > 0)

            to_add = {}
            for s in b:
                links = relationship_know.get(s, utils.getSemanticRelatioshipBabelnet(s, BABEL_KEY))
                to_add.update({s: links})

            d.update({l: to_add})

            with open(out_file, 'w') as f:
                f.write(json.dumps(d, ensure_ascii=False))
        else:
            print(l, 'already present')

    with open(out_file, 'w') as f:
        f.write(json.dumps(d, ensure_ascii=False))

    return d


def disambiguate_file(g, file, synsets_relationship):

    path, out_file = file.rsplit('/', 1)
    out_file_pickle = os.path.join(path, 'dis_'+out_file+'.json')
    out_file = os.path.join(path, 'dis_'+out_file)

    print(out_file)
    all_keys = []
    all_sent = []

    with open(file) as f:
        s = []
        for line in tqdm.tqdm(f):
            l = line.strip()
            tokens = l.split()

            if len(l) == 0:

                idx = [i for i, w in enumerate(s) if w[1] != '_']
                keys = [s[i][1] for i in idx]

                all_sent.append(s)
                all_keys.append(keys)

                s = []
                continue

            lemma = tokens[2]
            pos = tokens[4]

            if tokens[12] == 'Y':
                if lemma == '%':
                    k = '%25_' + mapping.map_tag('wsj', 'universal', pos)
                else:
                    k = lemma + '_' + mapping.map_tag('wsj', 'universal', pos)
                s.append((lemma, k))
            else:
                s.append((lemma, '_'))

    dis = graph.staticPagerankPrediction(g, all_keys, synsets_relationship)
    for i, s in enumerate(all_sent):
        keys = iter(dis[i])
        idx = [j for j, w in enumerate(s) if w[1] != '_']

        for i in range(len(s)):
            if i in idx:
                s[i] = s[i][0] + ' ' + next(keys)
            else:
                s[i] = s[i][0] + ' ' + '_'

        with open(out_file, "a") as f:
            for l in s:
                f.write(l + '\n')
            f.write('\n')

    with open(out_file_pickle, 'w') as f:
        f.write(json.dumps(all_sent, ensure_ascii=False))


train = utils.getTrainDataset(corpus='./dataset/semcor.data.xml', keysfile='./dataset/semcor.gold.key.bnids.txt')
testset = utils.getTrainDataset('./dataset/ALL.data.xml', './dataset/ALL.gold.key.bnids.txt')
for k in testset.keys():
    for i, v in enumerate(testset[k]):
        train.update({k+str(i): v})


train_rel, _ = utils.getSemanticRelationships(file='./dataset/data_train.json',
                                                   keyFile='../semcor.gold.key.bnids.txt', limit=0)
eval_rel, _ = utils.getSemanticRelationships(file='./dataset/data_eval_WN.json',
                                                   keyFile='./dataset/ALL.gold.key.bnids.txt', limit=0)

train_rel.update(eval_rel)

files = [
    '../../SRLData/EN/CoNLL2009-ST-English-development.txt',
                 '../../SRLData/EN/CoNLL2009-ST-English-train.txt',
                 '../../TestData/test.csv',
                #'../../TestData/testverbs.csv'
         ]

lemmas = {}

with open('./dataset/test_set.json', 'r') as f:
    data = f.read()
    d = json.loads(data)
    lemmas.update(d)

with open('./dataset/test_set.json', 'r') as f:
    data = f.read()
    d = json.loads(data)
    lemmas.update(d)

j = generate_dataset_json(files, './dataset/collnn_syns_pos.json', lemmas, train_rel)

G = graph.createGraph(semantic_relationships=[], graph_file='./dataset/dis_graph.adjlist')

for f in files:
    disambiguate_file(G, f, j)