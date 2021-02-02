import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, word_tokenize
import os
import numpy as np
import re
import json
# import math

train_path = os.path.join("./data/train.csv")
alphabetic_only = re.compile('[^a-zA-Z ]')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

big_dict = dict({"__general__": {}})
target_dict = dict({"__general__": {}})
non_target_dict = dict({"__general__": {}})

histogram_path = 'histogram.json'
tfidf_path = 'tfidf.json'

to_print_ids = set([85, 86, 87, 88, 89, 90])


def get_histogram():

    df = pd.read_csv(train_path)

    df.loc[:, "text"] = df.text.apply(lambda x: pre_processing(x))
    df.loc[:, "keyword"] = df.keyword.apply(
        lambda x: pre_processing(x, stemming=True))

    types = list(df.keyword.unique())
    # print(types)
    # print(len(types))
    for t in types:
        # print(t)
        # if pd.isna(t):
        #     t = "__empty__"

        type_frame = df[(df.keyword == t)]
        target_frame = type_frame[(type_frame.target == 1)]
        non_target_frame = type_frame[(type_frame.target == 0)]

        target_histogram = dict()
        non_target_histogram = dict()

        target_frame.text.apply(
            lambda x: count_word_in_text(x, target_histogram))
        non_target_frame.text.apply(
            lambda x: count_word_in_text(x, non_target_histogram))

        target_dict[t] = dict()
        non_target_dict[t] = dict()
        big_dict[t] = dict()

        for word, count in target_histogram.items():
            big_dict['__general__'][word] = big_dict['__general__'].setdefault(
                word, 0) + count
            target_dict['__general__'][word] = target_dict['__general__'].setdefault(
                word, 0) + count
            if count > 1:
                big_dict[t][word] = count
                target_dict[t][word] = count

        for word, count in non_target_histogram.items():
            big_dict['__general__'][word] = big_dict['__general__'].setdefault(
                word, 0) + count
            non_target_dict['__general__'][word] = non_target_dict['__general__'].setdefault(
                word, 0) + count
            if count > 1:
                big_dict[t][word] = count
                non_target_dict[t][word] = count

        # print(target_dict[t])

        # if t == "__empty__":
        #     print(target_frame)
        #     print(non_target_frame)
        big_dict['__general__'] = {
            word: count for word, count in big_dict['__general__'].items() if count > 1}
        target_dict['__general__'] = {
            word: count for word, count in big_dict['__general__'].items() if count > 1}
        non_target_dict['__general__'] = {
            word: count for word, count in big_dict['__general__'].items() if count > 1}
        json_content = {'all': big_dict, 'target': target_dict,
                        'non_target': non_target_dict}

        with open(histogram_path, 'w') as file:
            json.dump(json_content, file)
        # print(df.loc[df['id'].isin(to_print_ids)])

        pass


def tf_idf():
    with open(histogram_path, 'r') as file:
        histogram = json.load(file)
    target_tfidf = dict()
    for t, hist in histogram['target'].items():
            target_tfidf[t] = dict()
        for word, count in hist.items():
            pass



def count_word_in_text(text, d):
    for word in text.split():
        if len(word) < 3:
            continue
        d[word] = d.setdefault(word, 0) + 1


def pre_processing(text, lemmatization=True, stemming=False):
    def lemmatize_w_tags(pair, stemming):
        def get_wordnet_pos(treebank_tag):
            """
            return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
            """
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                # As default pos in lemmatization is Noun
                return wordnet.NOUN
        w = pair[0]
        t = get_wordnet_pos(pair[1])
        result = lemmatizer.lemmatize(w, t)

        return result if not stemming else stemmer.stem(result)

    if pd.isna(text):
        return "__empty__"

    # lower'em
    text = text.lower()

    # remove all non-alphabetic chars
    text = re.sub('%20', ' ', text)
    text = alphabetic_only.sub(' ', text)

    # remove stop words & extra white spaces
    # lemmatize, not stem

    tokens = [lemmatize_w_tags(p, stemming)
              for p in pos_tag(word_tokenize(text)) if p[0] not in stop_words]
    text = ' '.join(tokens)
    # text = lemmatizer.lemmatize(text)

    return text


# targets tend to have real locations.
def location_predictor(text):

    return False


def test():
    # from nltk.stem import WordNetLemmatizer

    # from pywsd.utils import lemmatize_sentence
    # n = WordNetLemmatizer()

    # print(n.lemmatize("blood"))
    # print(n.lemmatize("bleeding", pos='v'))
    # print(n.lemmatize("bled", pos='v'))
    # print(n.lemmatize("bleed", pos='v'))
    # print(n.lemmatize("blooding"))
    # print(n.lemmatize("bloody", pos='a'))
    # print(n.lemmatize("bloodiest", pos='a'))
    # print(n.lemmatize("bloodier", pos='a'))
    sentence = "Hello my name is Derek. I live in Salt Lake city. bloody hell who made this thing"
    sentence = "320 [IR] ICEMOON [AFTERSHOCK] | http://t.co/M4JDZMGJoW | @djicemoon | #Dubstep #TrapMusic #DnB #EDM #Dance #IcesÛ_ http://t.co/n0uhAsfkBv"

    sentence = sentence.lower()
    sentence = re.sub('%20', ' ', sentence)
    sentence = alphabetic_only.sub(' ', sentence)

    # print(' '.join([lemmatize_w_tags(p)
    #                 for p in pos_tag(word_tokenize(sentence))]))


if __name__ == '__main__':
    # get_histogram()
    # test()
    tf_idf()
