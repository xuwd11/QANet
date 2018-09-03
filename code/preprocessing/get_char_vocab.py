'''Get character vocabulary from dataset'''

import os
import argparse
import json
from collections import Counter
from tqdm import tqdm

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_name', required=True)
    return parser.parse_args()
    
def get_vocab(data_file, out_file):
    print('Open and count characters in {} ...'.format(data_file))
    with open(data_file, 'r') as f:
        char_counter = Counter(list(f.read()))
    print('Character vocabulary size: {}'.format(len(char_counter)))
    print('Save character vocabulary to {} ...'.format(out_file))
    with open(out_file, 'w') as fout:
        json.dump(char_counter, fout)
    print('Character vocabulary successfully saved.')

def main():
    args = setup_args()
    
    data_file = os.path.join(args.data_dir, args.data_name)
    out_file = os.path.join(args.data_dir, 'char_vocab.json')
    
    get_vocab(data_file, out_file)

if __name__ == '__main__':
    main()