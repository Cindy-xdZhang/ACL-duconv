#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

import re
import os
import random
import numpy as np

#TODO：载入知识的核心
class KnowledgeCorpus(object):
    """ Corpus """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 vocab_path,
                 min_len,
                 max_len):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.vocab_path = vocab_path
        self.min_len = min_len
        self.max_len = max_len

        self.current_train_example = -1
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.load_voc()

        def filter_pred(ids): 
            """
            src_filter_pred
            """
            return self.min_len <= len(ids) <= max_len
        self.filter_pred = lambda ex: filter_pred(ex['src']) and filter_pred(ex['tgt'])

    def load_voc(self):
        """ load vocabulary """
        idx = 0~
        self.vocab_dict = dict()
        with open(self.vocab_path, 'r') as fr: 
            for line in fr: 
                line = line.strip()
                self.vocab_dict[line] = idx
                idx += 1
    #读取stream record file，返回train_raw =【{'src': src, 'tgt': tgt, 'cue':filter_knowledge}....】元素为字典的列表
    def read_data(self, data_file):
        """ read_data """
        data = []
        with open(data_file, "r") as f:
            for line in f: 
                if line.rstrip('\n').split('\t') < 3: 
                    continue
                src, tgt, knowledge = line.rstrip('\n').split('\t')[:3]
                filter_knowledge = []
                for sent in knowledge.split('\1'):
                    #sent为一条三元组
                    filter_knowledge.append(' '.join(sent.split()[: self.max_len]))
                data.append({'src': src, 'tgt': tgt, 'cue':filter_knowledge})
        return data
    #tokenize数字并根据词汇表把src\tgt\cue文本串（chatpath+knowledge+":"+history\ response\ KG cue）转为数字串
    def tokenize(self, tokens): 
        """ map tokens to ids """
        if isinstance(tokens, str): 
            tokens = re.sub('\d+', '<num>', tokens).lower()
            toks = tokens.split(' ')
            toks_ids = [self.vocab_dict.get('<bos>')] + \
                       [self.vocab_dict.get(tok, self.vocab_dict.get('<unk>'))
                                for tok in toks] + \
                       [self.vocab_dict.get('<eos>')]
            return toks_ids
        elif isinstance(tokens, list):
            tokens_list = [self.tokenize(t) for t in tokens]
            return tokens_list
    #处理读取stream record file得到的train_raw =【{'src': src, 'tgt': tgt, 'cue':filter_knowledge}....】元素为字典的列表
    #处理包含对读入的文本进行tokennize；根据max min len 进行过滤
    #并根据词汇表把src\tgt\cue文本串（chatpath+knowledge+":"+history\ response\ KG cue）转为数字串
    #返回的examples为一个列表，列表每一项为(example['src'], example['tgt'], example['cue'])元组
    def build_examples(self, data):
        """ build examples, data: ``List[Dict]`` """
        examples = []
        for raw_data in data:
            #raw_data 为一个{'src': src, 'tgt': tgt, 'cue':filter_knowledge}字典
            example = {}
            for name, strings in raw_data.items():
                #name 为标签（src\tgt\cue) ; strings 为汉字文本内容（chatpath+knowledge+":"+history\ response\ KG cue）
                example[name] = self.tokenize(strings)
            #验证src 和tgt的长度在min_len <= len(ids) <= max_len
            if not self.filter_pred(example): 
                continue
            examples.append((example['src'], example['tgt'], example['cue']))
        return examples

    def preprocessing_for_lines(self, lines, batch_size):
        """ preprocessing for lines """
        raw_data = []
        for line in lines:
            src, tgt, knowledge = line.rstrip('\n').split('\t')[:3]
            filter_knowledge = []
            for sent in knowledge.split('\1'):
                filter_knowledge.append(' '.join(sent.split()[: self.max_len]))
            raw_data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge})

        examples = self.build_examples(raw_data)

        def instance_reader():
            """ instance reader """
            for (index, example) in enumerate(examples):
                instance = [example[0], example[1], example[2]]
                yield instance

        def batch_reader(reader, batch_size):
            """ batch reader """
            batch = []
            for instance in reader():
                if len(batch) < batch_size:
                    batch.append(instance)
                else:
                    yield batch
                    batch = [instance]

            if len(batch) > 0:
                yield batch

        def wrapper():
            """ wrapper """
            for batch in batch_reader(instance_reader, batch_size):
                batch_data = self.prepare_batch_data(batch)
                yield batch_data

        return wrapper

    def data_generator(self, batch_size, phase, shuffle=False): 
        """ Generate data for train, dev or test. """
        if phase == 'train':
            #读取stream record file
            train_file = os.path.join(self.data_dir, self.data_prefix + ".train")
            #返回train_raw =【{'src': src, 'tgt': tgt, 'cue':filter_knowledge}....】元素为字典的列表
            train_raw = self.read_data(train_file)
            #build_examples：
            #处理读取stream record file得到的train_raw =【{'src': src, 'tgt': tgt, 'cue':filter_knowledge}....】元素为字典的列表
            #处理包含对读入的文本进行tokennize；根据max min len 进行过滤
            #并根据词汇表把src\tgt\cue文本串（chatpath+knowledge+":"+history\ response\ KG cue）转为数字串
            #返回的examples为一个列表，列表每一项为(example['src'], example['tgt'], example['cue'])元组
            examples = self.build_examples(train_raw)
            self.num_examples['train'] = len(examples)
        elif phase == 'dev': 
            valid_file = os.path.join(self.data_dir, self.data_prefix + ".dev")
            valid_raw = self.read_data(valid_file)
            examples = self.build_examples(valid_raw)
            self.num_examples['dev'] = len(examples)
        elif phase == 'test': 
            test_file = os.path.join(self.data_dir, self.data_prefix + ".test")
            test_raw = self.read_data(test_file)
            examples = self.build_examples(test_raw)
            self.num_examples['test'] = len(examples)
        else: 
            raise ValueError(
                    "Unknown phase, which should be in ['train', 'dev', 'test'].")
        
        def instance_reader():
            """ instance reader """
            if shuffle: 
                random.shuffle(examples)
            for (index, example) in enumerate(examples): 
                #example为元组(example['src'], example['tgt'], example['cue'])
                #paddlepaddle规定reader读到的内容为可地带容器，且基本元素必须是元组
                if phase == 'train': 
                    self.current_train_example = index + 1
                instance = [example[0], example[1], example[2]]
                yield instance

        def batch_reader(reader, batch_size):
            """ batch reader """
            batch = []
            for instance in reader(): 
                if len(batch) < batch_size: 
                    batch.append(instance)
                else: 
                    yield batch
                    batch = [instance]

            if len(batch) > 0:
                yield batch

        def wrapper():
            """ wrapper """
            for batch in batch_reader(instance_reader, batch_size): 
                #进行padding并返回padding后的串和每个串的原长
                batch_data = self.prepare_batch_data(batch)
                yield batch_data

        return wrapper
    #进行padding并返回padding后的串和每个串的原长
    def prepare_batch_data(self, batch): 
        """ generate input tensor data """
        #ids表示经过词汇表转化的数字串
        batch_source_ids = [inst[0] for inst in batch]
        batch_target_ids = [inst[1] for inst in batch]
        batch_knowledge_ids = [inst[2] for inst in batch]
        #计算最大长度
        pad_source = max([self.cal_max_len(s_inst) for s_inst in batch_source_ids])
        pad_target = max([self.cal_max_len(t_inst) for t_inst in batch_target_ids])
        pad_kn = max([self.cal_max_len(k_inst) for k_inst in batch_knowledge_ids])
        pad_kn_num = max([len(k_inst) for k_inst in batch_knowledge_ids])
        #pad
        source_pad_ids = [self.pad_data(s_inst, pad_source) for s_inst in batch_source_ids]
        target_pad_ids = [self.pad_data(t_inst, pad_target) for t_inst in batch_target_ids]
        knowledge_pad_ids = [self.pad_data(k_inst, pad_kn, pad_kn_num) for k_inst in batch_knowledge_ids]

        source_len = [len(inst) for inst in batch_source_ids]
        target_len = [len(inst) for inst in batch_target_ids]
        kn_len = [[len(term) for term in inst] for inst in batch_knowledge_ids]
        kn_len_pad = []
        for elem in kn_len: 
            if len(elem) < pad_kn_num: 
                elem += [self.vocab_dict['<pad>']] * (pad_kn_num - len(elem))
            kn_len_pad.extend(elem)
        
        return_array = [np.array(source_pad_ids).reshape(-1, pad_source), np.array(source_len),
                        np.array(target_pad_ids).reshape(-1, pad_target), np.array(target_len),
                        np.array(knowledge_pad_ids).astype("int64").reshape(-1, pad_kn_num, pad_kn),
                        np.array(kn_len_pad).astype("int64").reshape(-1, pad_kn_num)]

        return return_array

    def pad_data(self, insts, pad_len, pad_num=-1): 
        """ padding ids """
        insts_pad = []
        if isinstance(insts[0], list): 
            for inst in insts: 
                inst_pad = inst + [self.vocab_dict['<pad>']] * (pad_len - len(inst))
                insts_pad.append(inst_pad)
            if len(insts_pad) < pad_num: 
                insts_pad += [[self.vocab_dict['<pad>']] * pad_len] * (pad_num - len(insts_pad))
        else: 
            insts_pad = insts + [self.vocab_dict['<pad>']] * (pad_len - len(insts))
        return insts_pad
    
    def cal_max_len(self, ids): 
        """ calculate max sequence length """
        if isinstance(ids[0], list): 
            pad_len = max([self.cal_max_len(k) for k in ids])
        else: 
            pad_len = len(ids)
        return pad_len
