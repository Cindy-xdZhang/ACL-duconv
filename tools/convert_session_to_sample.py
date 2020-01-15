#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: convert_session_to_sample.py
对每一行（每一条记录）执行如下代码，把一次多轮对话拆成多段历史回复对话，注意response永远认为是偶数下标句 也就是对话的奇数句子(第1，3，5，7句)
写出到sample.train.txt
"""

from __future__ import print_function
# from importlib import reload
import sys
import json
import collections

reload(sys)
sys.setdefaultencoding('utf8')
#python3 默认使用utf-8,不再需要修改系统了


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w')
    with open(session_file, 'r') as f:
        #每一行也是一条独立的记录
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", \
                                      object_pairs_hook=collections.OrderedDict)
            conversation = session["conversation"]

            for j in range(0, len(conversation), 2):
                sample = collections.OrderedDict()
                sample["goal"] = session["goal"]
                sample["knowledge"] = session["knowledge"]
                sample["history"] = conversation[:j]
                sample["response"] = conversation[j]

                sample = json.dumps(sample, ensure_ascii=False, encoding="utf-8")

                fout.write(sample + "\n")

    fout.close()


def main():
    """
    main
    """
    convert_session_to_sample(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
