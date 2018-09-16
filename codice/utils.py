import collections
import numpy as np
import zipfile
from gensim.models import KeyedVectors
import pickle
import string
from collections import  defaultdict
import os


WORD_DROP_SEED = 10
np.random.seed(12)
PUNT = list(string.punctuation)
PUNT.extend(["''", "``", "-lrb-", "-rrb-"])


class Dataset_nasari:
    '''
    This is the main dataset class. It builds and manage the dataset to be passed to the network.
    '''
    def __init__(self, train_file, train_synsets, develop_file=None, develop_synset=None, w2v_emb=None, nasari_emb=None,
                 batch_size=32, wd=0.25):

        self.wd = wd
        self.training = False

        self.batch_size = batch_size

        self.train_synsets = train_synsets

        self.word_id_map, self.pos_id_map, self.role_id_map, self.synset_id_map, self.word_count =\
            self._get_dicts(train_file)

        self.id_word_map = {v: k for k, v in self.word_id_map.items()}
        self.id_synset_map = {v: k for k, v in self.synset_id_map.items()}

        self.roles_len = len(self.role_id_map)
        self.pos_len = len(self.pos_id_map)
        self.words_len = len(self.word_id_map)
        self.lemma_len = len(self.synset_id_map)
        self.minlen = 10
        self.maxlen = None

        with open(w2v_emb, 'rb') as f:
            self.w2v = pickle.load(f)
        w2v_map = {}
        for w in self.w2v:
            w2v_map.update({w: len(w2v_map)})
        self.w2v_map = w2v_map

        with open(nasari_emb, 'rb') as f:
            self.nasari = pickle.load(f)
        nasari_map = {}
        for w in self.nasari:
            nasari_map.update({w: len(nasari_map)})
        self.nasari_map = nasari_map

        self.dataset, self.train_indexes = self._create_dataset(train_file, self.train_synsets)
        self.train_idx = list(np.arange(len(self.train_indexes)))
        np.random.shuffle(self.train_idx)

        self.develop_dataset, self.dev_indexes = self._create_dataset(develop_file, develop_synset)

    def _get_dicts(self, file):
        '''
        Create the dictionaryes to map values to indexes. It is used only one time on train dataset
        '''
        words = list()
        predicates = set()
        pos = set()
        roles = set()
        synsets_set = set()
        synsets_set.add('UNK')
        sd = {}

        with open(self.train_synsets, 'r') as ts:
            for l in ts:
                l = l.strip()
                if len(l) == 0:
                    continue
                _, syn = l.split()
                if syn != '_':
                    synsets_set.add(syn)

        for s in list(sorted(synsets_set)):
            sd.update({s: len(sd)})

        with open(file) as f:
            for l in f:
                line = l.strip()
                if len(line) == 0:
                    continue
                else:
                    tokens = line.split()
                    pos.add(tokens[4])

                    word = tokens[2]

                    try:
                        _ = float(word)
                        word = 'numeric'
                    except ValueError:
                        pass
                    words.append(word)

                    if tokens[12] == 'Y':
                        predicates.add(tokens[2])

                    r = tokens[14:]

                    for i in np.arange(len(r)):
                        if r[i] == '_':
                            r[i] = 'NULL'
                    roles.update(r)

        pos = sorted(pos)
        pd = {}
        for p in pos:
            pd.update({p: len(pd)})

        rd = {}
        roles = sorted(roles)
        for r in roles:
            rd.update({r: len(rd)})

        mc = collections.Counter(words)
        ld = {'UNK': 0}

        for w, c in mc.items():
            word = w
            try:
                _ = float(w)
                word = 'numeric'
            except ValueError:
                pass
            if word not in ld:
                ld.update({word: len(ld)})

        return ld, pd, rd, sd, mc

    def get_w2v_embeddings(self):
        m = np.empty((len(self.w2v), 300))
        for i, (w, _) in enumerate(self.w2v_map.items()):
            m[i] = self.w2v.get(w)
        return m

    def get_nasari_embeddings(self):
        m = np.empty((len(self.nasari), 300))
        for i, (w, _) in enumerate(self.nasari_map.items()):
            if i == 0:
                continue
            m[i-1] = self.nasari.get(w)
        return m

    def _create_dataset(self, file, synset_file):
        '''
        Given a file and its associated disambiguated file build the dataset
        '''
        sentences = []
        indexes = []
        all_roles = []

        sentence_i = 0
        line_i = 0

        synsets_file = open(synset_file, 'r')

        with open(file) as f:
            sent = []
            for l in f:
                s_line = next(synsets_file).strip()
                line = l.strip()
                if len(line) == 0:
                    if len(sent) > 0:
                        matrix_sent = self._process_sentence(sent)
                        predicates = np.arange(len(matrix_sent))[matrix_sent[:, 3] == 1]
                        indexes.extend([(sentence_i, p, matrix_sent.shape[0]) for p in predicates])

                        sentences.append(matrix_sent)
                        sentence_i += 1

                    line_i = 0
                    sent = []
                else:
                    tokens = line.split()
                    word = tokens[2]
                    try:
                        _ = float(word)
                        word = 'numeric'
                    except ValueError:
                        pass

                    pos = tokens[4]
                    head = str(int(tokens[8])-1) if int(tokens[8]) > 0 else '-1'
                    is_predicate = tokens[12]
                    roles = tokens[14:]
                    synset = s_line.split()[1]

                    sentence = {'word': word, 'pos': pos, 'head': head, 'is_predicate': is_predicate, 'roles': roles,
                                'n_preds': len(roles)+1, 'synset': synset}
                    line_i += 1
                    all_roles.extend(roles)
                    sent.append(sentence)

        if len(sent) > 0:
            matrix_sent = self._process_sentence(sent)
            predicates = np.arange(len(matrix_sent))[matrix_sent[:, 3] == 1]
            indexes.extend([(sentence_i, p, matrix_sent.shape[0]) for p in predicates])
            sentences.append(matrix_sent)

        synsets_file.close()
        return np.asarray(sentences), indexes

    def _process_sentence(self, sentence):
        sent = np.zeros((len(sentence), sentence[0]['n_preds'] + 5), dtype=int)
        nr, nc = sent.shape

        for i in np.arange(nr):
            s = sentence[i]

            sent[i][0] = self.word_id_map.get(s['word'], self.word_id_map['UNK'])
            sent[i][1] = self.pos_id_map[s['pos']]
            sent[i][2] = int(s['head'])

            pred = s['is_predicate'] == 'Y'
            sent[i][3] = 1 if pred else 0

            if pred:
                sent[i][4] = np.count_nonzero(sent[:, 3]) + 5

            synset = self.synset_id_map['UNK']
            if s['synset'] != '_' and s['synset'] in self.synset_id_map:
                synset = self.synset_id_map[s['synset']]

            sent[i][5] = synset

            roles = iter(s['roles'])

            if len(s['roles']) > 0:
                for j in np.arange(6, len(sent[i])):
                    com = next(roles)
                    if com == '_':
                        role = self.role_id_map['NULL']
                    else:
                        role = self.role_id_map.get(com, 'UNK')
                    assert (role != 'UNK')

                    sent[i][j] = role

        return sent

    def process_index(self, i, data_matrix=None, predicate_i=None):
        '''
        Process a given index to extract the single data in the batch
        '''
        r, c = data_matrix.shape
        words_ids = data_matrix[:, 0]
        pos_ids = data_matrix[:, 1]
        lemma_ids = data_matrix[:, 5]

        predicate = np.zeros((r, 1), dtype=np.uint8)
        predicate[predicate_i] = 1
        shape = data_matrix.shape

        if shape[1]-1 < data_matrix[predicate_i, 4]:
            col = np.empty(shape[0])
            col.fill(self.role_id_map['NULL'])
        else:
            col = data_matrix[:, data_matrix[predicate_i, 4]]

        word_emb = np.zeros(r, dtype=int)

        pos_emb = np.zeros(r, dtype=int)

        syns_emb = np.zeros(r, dtype=int)

        for j in np.arange(r):
            if self.training:
                prob = self.wd / (self.wd + self.word_count[self.id_word_map[words_ids[j]]]) + np.random.uniform()
                mask = 1 - np.floor(prob)
            else:
                mask = 1

            word_emb[j] = words_ids[j] * mask
            pos_emb[j] = pos_ids[j]
            syns_emb[j] = lemma_ids[j]

        syns_emb *= np.squeeze(predicate)

        w2v = np.zeros(r, dtype=int)
        for j in np.arange(r):
            lemma = self.id_word_map[words_ids[j]]
            l = self.w2v_map['UNK']
            if lemma in self.w2v_map:
                l = self.w2v_map.get(lemma)
            w2v[j] = l

        nasari = np.zeros(r, dtype=int)
        for j in np.arange(r):
            synset = self.id_synset_map[lemma_ids[j]]
            s = self.nasari_map['UNK']
            if synset in self.nasari_map:
                s = self.nasari_map.get(synset)
            nasari[j] = s
        nasari *= np.squeeze(predicate)

        return data_matrix, word_emb, pos_emb, predicate_i, col, w2v, nasari

    def get_batch(self, idx, indexes, dataset):
        '''
        create and returns a batch given a dataset and a set of indexes
        '''
        data_lengths = []
        w_emb = []
        p_emb = []
        predicates = []
        all_y = []
        w2vs = []
        lemma_emb = []

        batch = len(idx)

        for i in idx:
            data_matrix_i, predicate_i, _ = indexes[i]
            data_matrix = dataset[data_matrix_i]

            m, word_emb, pos_emb, predicate_i, col, w2v, syns = self.process_index(i, data_matrix=data_matrix,
                                                                                   predicate_i=predicate_i)
            data_lengths.append(m.shape[0])
            w_emb.append(word_emb)
            p_emb.append(pos_emb)
            predicates.append(predicate_i)
            all_y.append(col)
            w2vs.append(w2v)
            lemma_emb.append(syns)

        m = np.max(data_lengths)
        w_embs = np.zeros((batch, m), dtype=int)
        p_embs = np.zeros((batch, m), dtype=int)
        l_embs = np.zeros((batch, m), dtype=int)
        w2v_emb = np.zeros((batch, m), dtype=int)

        pred_idxs = np.zeros((batch, ))
        ys = np.zeros((batch, m), dtype=np.uint16)

        for i in np.arange(batch):
            d = data_lengths[i]
            w_embs[i, :d] = w_emb[i]
            p_embs[i, :d] = p_emb[i]
            l_embs[i, :d] = lemma_emb[i]
            pred_idxs[i] = predicates[i]
            ys[i, :d] = all_y[i]
            w2v_emb[i, :d] = w2vs[i]

        return w_embs, p_embs, pred_idxs, ys, np.asarray(data_lengths), w2v_emb, l_embs

    def get_train_batch(self, batch=None):
        start = 0
        if batch is None:
            batch = self.batch_size
        if batch is None:
            batch = self.batch_size

        end = min(len(self.train_idx), batch)

        indexes = self.train_indexes
        dataset = self.dataset

        while True:
            if start == len(self.train_idx):
                np.random.shuffle(self.train_idx)
                return

            idx = self.train_idx[start:end]
            yield self.get_batch(idx, indexes, dataset)

            start = end
            end += batch
            end = min(end, len(self.train_idx))

    def get_train_batch_incremental(self, batch=None):

        if batch is None:
            batch = self.batch_size

        indexes = self.train_idx.copy()
        np.random.shuffle(self.train_idx)

        first = None
        batches = []

        idxs = self.train_indexes
        dataset = self.dataset
        min_len = 0
        max_len = 14

        while len(indexes) > 0:
            if first is None:
                first = self.train_indexes[indexes[0]][2]
                min_len = first - 7
                max_len = first + 7
                b = []
            else:
                min_len -= 1
                max_len += 1

            for i in indexes:
                if min_len <= self.train_indexes[i][2] <= max_len and i not in b:
                    b.append(i)

                if len(b) == batch:
                    batches.append(b)
                    for i in b:
                        indexes.remove(i)
                    first = None
                    break

            if len(indexes) < batch:
                batches.append(b)
                break

        for b in batches:
            yield self.get_batch(b, idxs, dataset)

    def get_eval_batch(self, batch=None):
        if batch is None:
            batch = self.batch_size
        indexes = self.dev_indexes
        dataset = self.develop_dataset

        idx = np.random.choice(np.arange(len(indexes)), batch)
        a = self.get_batch(idx, dataset=dataset, indexes=indexes)
        return a

    def get_all_dataset(self, dataset=None, batch=None):
        start = 0
        if batch is None:
            batch = self.batch_size

        if isinstance(dataset, str):
            if dataset == 'dev':
                idxs = self.dev_indexes
                d = self.develop_dataset
            elif dataset == 'train':
                idxs = self.train_indexes
                d = self.dataset
        elif isinstance(dataset, tuple or list):
            d = dataset[0]
            idxs = dataset[1]
        else:
            assert False

        end = min(len(idxs), batch)
        idx_numbers = np.arange(len(idxs))

        while True:
            if start == len(idxs):
                return
            idx = idx_numbers[start:end]
            yield self.get_batch(idx, dataset=d, indexes=idxs)

            start = end
            end += batch
            end = min(end, len(idxs))

    def evaluate_test_data(self, model, path, dis_path):
        '''
        disambiguate the test dataset
        '''
        sentences, indexes = self._create_dataset(path, dis_path)
        sentences_string = []
        out_file = os.path.join(os.path.dirname(path), 'out.txt')

        with open(path) as f:
            com = []
            for l in f:
                line = l.strip()
                if len(line) == 0:
                    sentences_string.append(com)
                    com = []
                else:
                    com.append(line.split('\t'))
        print('Predictions...')
        d = defaultdict(list)
        model.load(best_one=True)

        for i, batch in enumerate(self.get_all_dataset(dataset=(sentences, indexes), batch=1)):
            predictions = model.forward(batch)
            d[indexes[i][0]].append(predictions[0])

        id_roles_map = {v: k for k, v in self.role_id_map.items()}
        f = lambda x: '_' if id_roles_map[x] == 'NULL' else id_roles_map[x]
        s = ''

        for i, str in enumerate(sentences_string):
            roles = None
            if i in d:
                roles = d.get(i)
                roles = [[f(r) for r in l] for l in roles]
                roles = list(zip(*roles))

            for l_n, line in enumerate(str):
                s += '\t'.join(line)
                if roles is not None:
                    s += '\t'
                    s += '\t'.join(roles[l_n])
                s += '\n'
            s += '\n'

        with open(out_file, "w") as f:
            f.write(s)


def extract_nasari(files, nasari_path, out_file='../embeddings/nasari.json'):
    '''
    Given a set of disambiguated files extract the nasari vectors
    '''
    d = dict()
    d.update({'UNK': np.random.RandomState(0).randn(300)})

    synsets = set()

    for f in files:
        with open(f, 'r') as file:
            for l in file:
                line = l.strip()
                if len(line) == 0:
                    continue
                else:
                    word, syn = line.split()
                    if syn != '_':
                        synsets.add(syn)
                        continue

    synsets = sorted(list(synsets))
    rnd = np.random.RandomState(10)
    d.update({'UNK': rnd.randn(300)})

    all_vectors = []

    with zipfile.ZipFile(nasari_path, 'r') as z:
        with z.open('NASARI_embed_english.txt', 'r') as f:
            next(f)
            for line in f:
                line = str(line)
                line = line.split(" ")
                s, v = line[0], line[1:]
                s = s.split("_", 1)[0][-12:]

                if s in synsets:
                    vector = np.fromstring(' '.join(v), sep=' ')
                    d.update({s: vector})
                    all_vectors.append(vector)
                    synsets.remove(s)

                if len(synsets) == 0:
                    break

    all_vectors = np.asarray(all_vectors)
    m, std = np.mean(all_vectors, axis=0), np.std(all_vectors, axis=0)
    for s in synsets:

        d.update({s: rnd.normal(loc=m, scale=std, size=300)})

    with open(out_file, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def extract_w2v(files, w2v_path, out_file='../embeddings/w2v.json'):
    '''
    given the datasets extract the w2v vectors
    '''
    lemmas = set()
    for f in files:
        with open(f, 'r') as file:
            for l in file:
                line = l.strip()
                if len(line) == 0:
                    continue
                else:
                    word = line.split()[2]

                    try:
                        _ = float(word)
                        word = 'numeric'
                    except ValueError:
                        pass
                    lemmas.add(word)

    goognews_wordecs = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    d = dict()
    d.update({'UNK': np.random.RandomState(1).randn(300)})
    unk = 0
    for l in lemmas:
        if l in goognews_wordecs:
            print(l)
            d.update({l: goognews_wordecs[l]})
            unk += 1
        else:
            print(l, 'non trovato')

    del goognews_wordecs

    with open(out_file, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)