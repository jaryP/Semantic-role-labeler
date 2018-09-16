import argparse
import sys
sys.path.append('..')
from codice.utils import Dataset_nasari
from codice.model import MLP
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='NLP HW3 main')

# /media/jary/DATA/Uni/NLP/17-18/HWs/HW3/SRLData/EN/CoNLL2009-ST-English-train.txt /media/jary/DATA/Uni/NLP/17-18/HWs/HW3/SRLData/EN/dis_CoNLL2009-ST-English-train.txt /media/jary/DATA/Uni/NLP/17-18/HWs/HW3/SRLData/EN/CoNLL2009-ST-English-development.txt '/media/jary/DATA/Uni/NLP/17-18/HWs/HW3/SRLData/EN/dis_CoNLL2009-ST-English-development.txt ../embeddings/w2v.json ../embeddings/nasari.json --model_json ./model.json,

parser.add_argument("--train_dataset",
                    metavar="<command>", type=str,
                    help='path of the train dataset')

parser.add_argument("--train_syn",
                    metavar="<command>", type=str,
                    help='path of the synset assocaited to the train dataset')

parser.add_argument("--dev_dataset",
                    metavar="<command>", type=str,
                    help='path of the dev dataset')

parser.add_argument("--dev_syn",
                    metavar="<command>", type=str,
                    help='path of the synset assocaited to the dev dataset')

parser.add_argument("--w2v_path",
                    metavar="<command>", type=str,
                    help='path of the synset assocaited to the dev dataset')

parser.add_argument("--nasari_path",
                    metavar="<command>", type=str,
                    help='path of the synset assocaited to the dev dataset')

parser.add_argument('--model_json', required=False,
                    default=None, type=str,
                    metavar="path to json containing model hyperparams and arguments",
                    help='Model json path')

parser.add_argument('--to_train', required=False,
                    default='y', type=str2bool,
                    metavar="path to json containing model hyperparams and arguments",
                    help='Model json path')

# parser.add_argument('--test_dataset', required=False,
#                     default=None, type=str,
#                     metavar="path to json containing model hyperparams and arguments",
#                     help='Model json path')
#
# parser.add_argument('--test_syn', required=False,
#                     default='y', type=str,
#                     metavar="path to json containing model hyperparams and arguments",
#                     help='Model json path')

args = parser.parse_args()

d = Dataset_nasari(train_file=args.train_dataset, train_synsets=args.train_syn,
                   w2v_emb=args.w2v_path, develop_file=args.dev_dataset, nasari_emb=args.nasari_path,
                   develop_synset=args.dev_syn)

model = {}
if args.model_json is not None:
    with open(args.model_json) as f:
        model = json.load(f)

lstm = MLP(dataset=d, **model)

if args.to_train:
    lstm.train()

lstm.study_results()
