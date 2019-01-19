import sys
import os
import pickle
import mmap
import re
from collections import Counter

punct_chars = "[-!\"#$%&'()*+,/:;<=>?@[\]^_`{|}~—»«“”„….]"

punct = re.compile(punct_chars)

def keep_hyphen(search_str, position):
    if search_str[position] != "-":
        return False
    if search_str[position] == "-" and \
            (position + 1 > len(search_str)):
        return False
    if search_str[position] == "-" and \
            (position + 1 < len(search_str)) and \
                (search_str[position + 1] in punct_chars or search_str[position + 1] == " "):
        return False
    return True

def expandall(sent_orig):
    pos = 0

    # replace illusive space
    sent = sent_orig.replace(" ", " ")
    sent = replace_accents_rus(sent)

    new_str = ""
    search_str = sent[0:]
    res = re.search(punct, search_str)

    while res is not None:
        begin_at = res.span()[0]
        end_at = res.span()[1]

        new_str += search_str[:begin_at]

        if len(new_str) > 0 and \
            begin_at != 0 and \
                search_str[begin_at] != "-" and \
                    new_str[-1] != " " and \
                        not keep_hyphen(search_str, begin_at): # some problem here << didn't detect --.
            new_str += " "
        new_str += search_str[begin_at]

        if len(search_str) > end_at and \
                search_str[begin_at] != "-" and \
                    search_str[end_at] != " ":
            new_str += " "

        if len(search_str) > end_at:
            search_str = search_str[end_at:]
        else:
            search_str = ""
        res = re.search(punct, search_str)
    new_str += search_str


    return new_str

def replace_accents_rus(sent_orig):

    sent = sent_orig.replace("о́", "о")
    sent = sent.replace("а́", "а")
    sent = sent.replace("е́", "е")
    sent = sent.replace("у́", "у")
    sent = sent.replace("и́", "и")
    sent = sent.replace("ы́", "ы")
    sent = sent.replace("э́", "э")
    sent = sent.replace("ю́", "ю")
    sent = sent.replace("я́", "я")
    sent = sent.replace("о̀", "о")
    sent = sent.replace("а̀", "а")
    sent = sent.replace("ѐ", "е")
    sent = sent.replace("у̀", "у")
    sent = sent.replace("ѝ", "и")
    sent = sent.replace("ы̀", "ы")
    sent = sent.replace("э̀", "э")
    sent = sent.replace("ю̀", "ю")
    sent = sent.replace("я̀", "я")
    # sent = sent.replace(b"\u0301".decode('utf8'), "")
    # sent = sent.replace(b"\u00AD".decode('utf8'), "")
    return sent


class Tokenizer:

    def __call__(self,lines):
        lines = lines.strip().split("\n")
        tokenized = ""
        for line in lines:
            tokenized += expandall(line.lower())
            # if len(lines) > 1:
            #     tokenized += " N "
        return tokenized.split()
