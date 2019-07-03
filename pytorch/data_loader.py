import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
import os

use_cuda = torch.cuda.is_available()

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, name_len, f_api, api_len, 
                 f_tokens, tok_len, f_descs=None, desc_len=None):
        self.data_dir=data_dir
        self.f_name=f_name
        self.name_len=name_len
        self.f_api=f_api
        self.api_len=api_len
        self.f_tokens=f_tokens
        self.tok_len=tok_len
        self.f_descs=f_descs
        self.desc_len=desc_len

        self.data_len=-1

        self.reader_dict=dict()
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            seq=np.append(seq, [PAD_token]*maxlen)
            seq=seq[:maxlen]
        else:
            seq=seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):         
        reader = None
        pid = os.getpid()
        if not pid in self.reader_dict:
            print(pid)
            reader=dict()   

            reader['training']=False
            print("loading data...")
            table_name = tables.open_file(self.data_dir+self.f_name)
            reader['names'] = table_name.get_node('/phrases')
            reader['idx_names'] = table_name.get_node('/indices')
            table_api = tables.open_file(self.data_dir+self.f_api)
            reader['apis'] = table_api.get_node('/phrases')
            reader['idx_apis'] = table_api.get_node('/indices')
            table_tokens = tables.open_file(self.data_dir+self.f_tokens)
            reader['tokens'] = table_tokens.get_node('/phrases')
            reader['idx_tokens'] = table_tokens.get_node('/indices')
            if self.f_descs is not None:
                reader['training']=True
                table_desc = tables.open_file(self.data_dir+self.f_descs)
                reader['descs'] = table_desc.get_node('/phrases')
                reader['idx_descs'] = table_desc.get_node('/indices')
            
            assert reader['idx_names'].shape[0] == reader['idx_apis'].shape[0]
            assert reader['idx_apis'].shape[0] == reader['idx_tokens'].shape[0]
            if self.f_descs is not None:
                assert reader['idx_names'].shape[0]==reader['idx_descs'].shape[0]
            reader['data_len'] = reader['idx_names'].shape[0]

            print("{} entries".format(reader['data_len']))
            self.reader_dict[pid]=reader
        else:
            reader=self.reader_dict[pid]

        len, pos = reader['idx_names'][offset]['length'], reader['idx_names'][offset]['pos']
        name = reader['names'][pos:pos + len].astype('int64')
        name = self.pad_seq(name, self.name_len)
        
        len, pos = reader['idx_apis'][offset]['length'], reader['idx_apis'][offset]['pos']
        apiseq = reader['apis'][pos:pos+len].astype('int64')
        apiseq = self.pad_seq(apiseq, self.api_len)

        len, pos = reader['idx_tokens'][offset]['length'], reader['idx_tokens'][offset]['pos']
        tokens = reader['tokens'][pos:pos+len].astype('int64')
        tokens = self.pad_seq(tokens, self.tok_len)

        if reader['training']:
            len, pos = reader['idx_descs'][offset]['length'], reader['idx_descs'][offset]['pos']
            good_desc = reader['descs'][pos:pos+len].astype('int64')
            good_desc = self.pad_seq(good_desc, self.desc_len)

            rand_offset=random.randint(0, reader['data_len']-1)
            len, pos = reader['idx_descs'][rand_offset]['length'], reader['idx_descs'][rand_offset]['pos']
            bad_desc = reader['descs'][pos:pos+len].astype('int64')
            bad_desc = self.pad_seq(bad_desc, self.desc_len)

            return name, apiseq, tokens, good_desc, bad_desc
        else:
            return name, apiseq, tokens
        
    def __len__(self):
        if self.data_len == -1:
            table_name = tables.open_file(self.data_dir+self.f_name)
            idx_names = table_name.get_node('/indices')
            self.data_len = idx_names.shape[0]
        return self.data_len

def load_dict(filename):
    #return json.loads(open(filename, "r").readline())
    return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    
    input_dir='./data/github/'
    VALID_FILE=input_dir+'train.h5'
    valid_set=CodeSearchDataset(VALID_FILE)
    valid_data_loader=torch.utils.data.DataLoader(dataset=valid_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    vocab = load_dict(input_dir+'vocab.json')
    ivocab = {v: k for k, v in vocab.items()}
    #print ivocab
    k=0
    for qapair in valid_data_loader:
        k+=1
        if k>20:
            break
        decoded_words=[]
        idx=qapair[0].numpy().tolist()[0]
        print (idx)
        for i in idx:
            decoded_words.append(ivocab[i])
        question = ' '.join(decoded_words)
        decoded_words=[]
        idx=qapair[1].numpy().tolist()[0]
        for i in idx:
            decoded_words.append(ivocab[i])
        answer=' '.join(decoded_words)
        print('<', question)
        print('>', answer)
