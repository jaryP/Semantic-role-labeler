import xml.etree.ElementTree as ET
import urllib.request as urllib2
import json
from collections import defaultdict
import os
import tqdm


class instance:
    def __init__(self, value, lemma, pos, key):
        self.value = value
        self.lemma = lemma
        self.pos = pos
        self.instance = key

    def __str__(self):
        return "value: {}, lemma: {}, pos: {}, instance {}".format(self.value, self.lemma, self.pos, self.instance)

    def __hash__(self):
        return hash(self.instance)

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.instance == other.instance


class wf:
    def __init__(self, value, lemma, pos):
        self.value = value
        self.lemma = lemma
        self.pos = pos

    def __str__(self):
        return "value: {}, lemma: {}, pos: {}".format(self.value, self.lemma, self.pos)


def getSynsetsDictionary(file):
    """
    Given a file containing the sense annotations this function return a set of synsets appearing in the file
    and a dictionary mapping each lemma to the correct sentence.

    :param file: path of the file
    :return: dictionary {lemma_id: correct_sense} and set of synsets
    """
    keyDict = dict()
    synsets = set()
    with open(file, 'r') as f:
        for line in f:
            s = line.split()
            keyDict[s[0]] = s[1]
            synsets.add(s[1])
    return keyDict, synsets


def getTrainDataset(corpus, keysfile):
    """
    This function build the train dataset as a dictioanry where a key is a document id and the respective value is a list
    containing all the words in that document
    :param corpus: path to the train file
    :param keysfile: path to the file containing the senses for each lemma in the train dataset.
    :return: a dictionary {document_id: [list of words in document]}
    """
    root = ET.parse(corpus).getroot()

    keydict, _ = getSynsetsDictionary(keysfile)

    documents = defaultdict(list)
    for neighbor in root.iter('sentence'):
        sentid = neighbor.get('id').split('.')
        sent = []
        for a in neighbor:
            if a.tag == 'wf':
                sent.append(wf(a.text, a.get('lemma'), a.get('pos')))
            else:
                sent.append(instance(a.text, a.get('lemma'), a.get('pos'), keydict[a.get('id')]))
        documents[sentid[0]].append(sent)

    return dict(documents)


def getEvalDataset(corpus, keysfile):
    '''
    This function build the evaluation dataset as a dictioanry where a key is the name of the evaluation dataset
    and the respective value is a dictioanry where a key is a document id and the respective value is a list
    containing all the words in that document
    :param corpus: path to the eval file
    :param keysfile: path to the file containing the senses for each lemma in the eval dataset.
    :return: a dictionary {eval_set: {document_id: [list of words in document}}
    '''

    root = ET.parse(corpus).getroot()

    keydict, _ = getSynsetsDictionary(keysfile)

    datasets = defaultdict(dict)

    for text in root.iter('text'):
        sentid = text.get('id').split('.')
        sent = []
        for neighbor in text.iter('sentence'):
            for a in neighbor:
                if a.tag == 'wf':
                    sent.append(wf(a.text, a.get('lemma'), a.get('pos')))
                else:
                    sent.append(instance(a.text, a.get('lemma'), a.get('pos'), keydict[a.get('id')]))
        datasets[sentid[0]].update({sentid[1]:sent})

    return datasets


def getDocumentsLemmas(document, eval=False):
    '''
    Give a list this function will return ambiguous words and ground truth synsets
    :param document: the document represented as a list of words
    :return: lemmas: list of lemmas, saved as lemma_POS-TAG, in the input and ground truth list associated to lemma list
    '''
    lemmas = []
    synsets = []

    if eval:
        document = [document]

    for sentence in document:
        for w in sentence:
            if isinstance(w, instance):
                lemmas.append(w.lemma+'_'+w.pos)
                synsets.append(w.instance)
    return lemmas, synsets


def getAssociatedSynsetsBabelnet(lemma, postag, key, wn=True, pos='pos'):

    '''
    For the given lemma and pos tag associated to the lemma returns all the associated synsets from WordNet
    :param lemma: the lemma
    :param postag: pos tag associated to the lemma
    :param key: Babelnet key
    :return: a list of synsets associated to the lemma
    '''

    if wn:
        ur = 'https://babelnet.io/v5/getSynsetIds?lemma={}&searchLang={}&{}={}&source=WN&key={}'.\
            format(lemma, 'EN', pos, postag, key)
    else:
        ur = 'https://babelnet.io/v5/getSynsetIds?lemma={}&searchLang={}&POS={}&key={}'.\
            format(lemma, 'EN', postag, key)

    request = urllib2.Request(ur)
    response = urllib2.urlopen(request)

    data = response.read()
    encoding = response.info().get_content_charset('utf-8')
    JSON_object = json.loads(data.decode(encoding))

    ids = []

    if 'message' in JSON_object:
        print(JSON_object)
        return -1

    for result in JSON_object:
        ids.append(result['id'])

    return ids


def getSemanticRelatioshipBabelnet(sid, key, allowedId = []):
    '''
    For a given synset returs all the outgoing semantic edges (based on BabelNet)
    :param sid: id of the synset
    :param key: Babelnet key
    :param allowedId: ids to include in the return
    :return: a dictionary {semantic_rel: [id semantically connected]}
    '''

    ur = 'https://babelnet.io/v5/getOutgoingEdges?id=' + sid + '&key=' + key
    request = urllib2.Request(ur)
    response = urllib2.urlopen(request)

    data = response.read()
    encoding = response.info().get_content_charset('utf-8')
    JSON_object = json.loads(data.decode(encoding))

    d = defaultdict(list)

    if 'message' in JSON_object:
        print(JSON_object)
        return -1

    for result in JSON_object:
        language = result.get('language')
        if language != 'EN':
            continue
        pointer = result['pointer']
        target = str(result.get('target'))
        if len(allowedId) != 0 and target not in allowedId:
            continue
        short = str(pointer.get('shortName'))
        d[short].append(target)

    return dict(d)


def getSemanticRelationships(file, keyFile, limit = -1):

    '''
    Give a file and the associated keyfile returns the connected synsets for each lemma in the file
    :param file: the file containing the corpus
    :param keyFile: the keyfile containing the correct synset for each ambiguous lemma
    :param limit: limit of synset to fetch from babelnet
    :return: dictionary {id: {semantic_rel: [id semantically connected]}} and the list of synsets
    '''

    babelkey = '32796f83-09b8-4c0f-8190-f57069a8f3cf'

    d = dict()
    if os.path.isfile(file):
        with open(file) as fp:
            data = fp.read()
            d = json.loads(data)

    keys = list(d.keys())

    if limit == 0:
        return d, keys

    _, synset = getSynsetsDictionary(keyFile)

    com = []
    for syn in synset:
        if syn not in keys:
            com.append(syn)
    i = 0
    for sis in tqdm.tqdm(com):
        try:
            v = getSemanticRelatioshipBabelnet(sis, babelkey)
        except Exception as e:
            print('\nError:', end=' ')
            print(e)
            continue
        if v == -1:
            continue
        d[sis] = v
        i+=1
        if i % 20 == 0:
            with open(file, 'w') as fp:
                data = json.dumps(d)
                fp.write(data)
        if i >= limit and limit != -1:
            break
    with open(file, 'w') as fp:
        data = json.dumps(d)
        fp.write(data)

    return d, list(d.keys())


def getAssociatedSynsets(file, testset=None, semantic_rel_know=dict(), limit=-1):

    '''
    Give a file returns, for each lemma, all the possible synsets and for each one all the semantic relationship
    :param file: file in which save the json
    :param testset: the testset
    :param semantic_rel_know: semantic association alredy know (to avoid useless babelnet request)
    :param limit: limit of synset to fetch from babelnet
    :return: a dictioanry {lemma: {semantic_rel: [id semantically connected]}}
    '''

    babelkey = '32796f83-09b8-4c0f-8190-f57069a8f3cf'

    d = dict()
    if os.path.isfile(file):
        with open(file) as fp:
            data = fp.read()
            d = json.loads(data)

    if limit == 0:
        return d, list(d.keys())

    assert (testset is not None)

    data = set()
    for key in testset.keys():
        for doc, sentences in testset[key].items():
            for w in sentences:
                if isinstance(w, instance):
                    tosearch_string = w.lemma.lower()+'_'+w.pos
                    if tosearch_string not in d:
                        data.add((w.lemma.lower(), w.pos, tosearch_string))
    i = 0
    for lemma in tqdm.tqdm(data):
        try:
            v = getAssociatedSynsetsBabelnet(lemma=lemma[0], postag=lemma[1], key=babelkey)
        except Exception as e:
            print('\nError:', end=' ')
            print(e)
            continue
        if v == -1:
            continue
        connections = {}
        for synset in v:
                connections[synset] = semantic_rel_know.get(synset, getSemanticRelatioshipBabelnet(synset, babelkey))
        d[lemma[2]] = connections
        i += 1
        if i % 20 == 0:
                with open(file, 'w') as fp:
                    fp.write(json.dumps(d))
        if i >= limit and limit != -1:
            break

    with open(file, 'w') as fp:
        fp.write(json.dumps(d))

    return d, list(d.keys())


def getTestDocuments(file):
    '''
    return a list of documents. Each document is formed by ambiguous lemmas
    :param file: the test file
    :return: a list of documents [[lemma_pro_id, ...],...]
    '''
    docs = []
    with open(file) as f:
        for l in f.readlines():
            val = l.split()
            doc = []
            for v in val:
                v = v.split('|')
                if len(v) != 4:
                    continue
                tosearch_string = v[1].lower() + '_' + v[2]+'_'+v[3]
                doc.append(tosearch_string)
            docs.append(doc)
    return docs


def getTestDataset(file, json_file, ret= False, semantic_rel_know=dict()):

    '''

    :param file: Test dataset
    :param json_file: json file containing the semantic relationship for each lemma in the test set
    :param ret: if the json shoul be returned or not
    :param semantic_rel_know: dictionary of alredy know semantic relationship
    :return: list of documents and dictionary {lamma: {semantic_rel:[synsets connected]}}
    '''

    babelkey = '32796f83-09b8-4c0f-8190-f57069a8f3cf'

    diz = dict()
    if os.path.isfile(json_file):
        with open(json_file) as fp:
            data = fp.read()
            diz = json.loads(data)

    docs = getTestDocuments(file)

    if ret:
        return docs, diz

    i = 0
    for doc in docs:
        for v in doc:
            lemma, pos, _ = v.rsplit('_')
            if lemma+'_'+pos not in diz:

                if lemma == '%':
                    lemma = '%25'
                try:
                    com = getAssociatedSynsetsBabelnet(lemma=lemma, postag=pos, key=babelkey)

                    if len(com) == 0:
                        com = getAssociatedSynsetsBabelnet(lemma=lemma, postag=pos, key=babelkey, wn=False)

                    if com == -1:
                        break

                    connections = {}

                    for synset in com:
                        connections[synset] = semantic_rel_know.get(synset, getSemanticRelatioshipBabelnet(synset, babelkey))

                    if lemma == '%25':
                        lemma = '%'

                    diz[lemma+'_'+pos] = connections

                    i += 1
                    if i % 20 == 0:
                        with open(json_file, 'w') as fp:
                            fp.write(json.dumps(diz))

                except Exception as e:
                    print(e)
                    print(lemma, pos)

    with open(json_file, 'w') as fp:
        fp.write(json.dumps(diz))

    return docs, diz