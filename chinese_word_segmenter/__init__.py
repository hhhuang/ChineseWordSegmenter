import pathlib
import io, zipfile
import os.path
import requests
import pandas as pd

import torch
from simpletransformers.ner import NERModel, NERArgs
from transformers import AutoTokenizer, AutoModel, utils, AutoModelForTokenClassification

__version__ = "0.1"

class ChineseWordSegmenter:
    def __init__(self):
        self.model_args = NERArgs()
        self.data_dir = str(pathlib.Path(__file__).parent.resolve())
        self.model_path = str(pathlib.Path(__file__).parent.resolve() / "trained_model")
        self.model_args.output_dir = os.path.join(self.model_path, "outputs")
        self.model_args.train_batch_size = 32
        self.model_args.evaluate_during_training = True
        self.model_args.labels_list = ["L", "M", "R", "S"]
        self.model_args.overwrite_output_dir = True
        self.model_args.silent = True
        self.model = None

    def compare(self, actual_toks, pred_toks):
        i = 0
        j = 0
        p = 0
        q = 0
        tp = 0
        fp = 0
        while i < len(actual_toks) and j < len(pred_toks):
            if p == q:
                if actual_toks[i] == pred_toks[j]:
                    tp += 1
                else:
                    fp += 1
                p += len(actual_toks[i])
                q += len(pred_toks[j])
                i += 1
                j += 1
            elif p < q:
                p += len(actual_toks[i])
                i += 1
            else:
                fp += 1
                q += len(pred_toks[j])
                j += 1
        return tp, fp, len(actual_toks)

    def score(self, actual_sents, pred_sents):
        print("Number of actual sents: ", len(actual_sents))
        print("Number of predicted sents: ", len(pred_sents))
        tp = 0
        fp = 0
        total = 0
        for actual_toks, pred_toks in zip(actual_sents, pred_sents):
            tp_, fp_, total_ = self.compare(actual_toks, pred_toks)
            tp += tp_
            fp += fp_
            total += total_
        recall = float(tp) / total
        precision = float(tp) / (tp + fp)
        f1 = 2.0 * recall * precision / (recall + precision)
        return recall, precision, f1

    def words_to_tags(self, words):
        tags = []
        for word in words:
            if len(word) == 1:
                tags.append('S')
            else:
                for i in range(len(word)):
                    if i == 0:
                        tags.append('L')
                    elif i == len(word) - 1:
                        tags.append('R')
                    else:
                        tags.append('M')
        return tags

    def download_data(self):
		#   Training/test data were provided by http://sighan.cs.uchicago.edu/bakeoff2005/
        remote_url = "https://raw.githubusercontent.com/hhhuang/nlp2019fall/master/word_segmentation/"
        r = requests.get(remote_url + "data/as_training.utf8", allow_redirects=True)
        open(os.path.join(self.data_dir, 'as_training.utf8'), 'wb').write(r.content)
        r = requests.get(remote_url + "data/as_testing_gold.utf8", allow_redirects=True)
        open(os.path.join(self.data_dir, 'as_testing_gold.utf8'), 'wb').write(r.content)

    def load_data(self):
        self.download_data()
        raw_train = []
        raw_test = []
        with open(os.path.join(self.data_dir, "as_training.utf8"), encoding="utf8") as fin:
            for line in fin:
                raw_train.append(line.strip().split("　"))   # It is a full white space
        with open(os.path.join(self.data_dir, "as_testing_gold.utf8"), encoding="utf8") as fin:
            for line in fin:
                raw_test.append(line.strip().split("　"))   # It is a full white space

        print("Number of sentences in the training data: %d" % len(raw_train))
        print("Number of sentences in the test data: %d" % len(raw_test))

        train_X = []
        train_Y = []

        test_X = []
        test_Y = []

        for sent in raw_train:
            train_X.append(list("".join(sent)))  # Make the unsegmented sentence as a sequence of characters
            train_Y.append(self.words_to_tags(sent))

        for sent in raw_test:
            test_X.append(list("".join(sent)))  # Make the unsegmented sentence
            test_Y.append(self.words_to_tags(sent))

        return train_X, train_Y, test_X, test_Y

    def prepare_data(self, X, Y, limit=None):
        data = []
        for sid, (x, y) in enumerate(zip(X, Y)):
            for x_tok, y_tok in zip(x, y):
                data.append([sid, x_tok, y_tok])
            if limit and sid >= limit:
                break
        return data

    def eval(self, test_data=None, size=None):
        self.download_data()
        if test_data is None:
            test_data = []
            with open(os.path.join(self.data_dir, "as_testing_gold.utf8"), encoding="utf8") as fin:
                for line in fin:
                    if size is not None and len(test_data) >= size:
                        break
                    test_data.append(line.strip().split("　"))   # It is a full white space
        elif size is not None:
            test_data = test_data[:size]
        
        print("Number of evaluation data: %d" % len(test_data))
        print(test_data[-10:])
        pred = []
        for s in ["".join(sent) for sent in test_data]:
            pred.append(self.tokenize(s))
        print(test_data[0], pred[0])
        return self.score(test_data, pred)

    def train(self):
        train_X, train_Y, test_X, test_Y = self.load_data()
        train_data = self.prepare_data(train_X, train_Y)
        eval_data = self.prepare_data(test_X, test_Y, 100)
        print(train_data[:10])
        print(eval_data[:10])
        train_data = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])
        eval_data = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])
        self.model_args.silent = False
        model = NERModel("bert", "bert-base-chinese", args=self.model_args, use_cuda = torch.cuda.is_available())
        model.train_model(train_data, eval_data=eval_data, output_dir=self.model_path)
        result, model_outputs, preds_list = model.eval_model(eval_data)
        #model.save_pretrained(self.model_path)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        if not os.path.isdir(self.model_args.output_dir):
            print("Starting to download trained model")
            with io.BytesIO() as content:
                for modelfile in ["model_aa", "model_ab", "model_ac", "model_ad", "model_ae", "model_af", "model_ag", "model_ah"]:
                    r = requests.get("https://github.com/hhhuang/ChineseWordSegmenter/raw/main/chinese_word_segmenter/trained_model/%s?download=" % modelfile)
                    print("%s is downloaded with %d" % (modelfile, len(r.content)))
                    content.write(r.content)
                z = zipfile.ZipFile(content)
                z.extractall(self.model_path)

        self.model = NERModel("bert", 
            self.model_args.output_dir, 
            args=self.model_args, 
            use_cuda=torch.cuda.is_available())

    def sent_tokenize(self, text):
        sents = []
        s = ""
        for ch in text:
            s += ch
            if ch in {'。', '，', '！', '？', '：', '；'}:
                sents.append(s)
                s = ""
        if s:
            sents.append(s)
        return sents

    def tokenize(self, text):
        if self.model is None:
            self.load_model()
        sents = [" ".join(list(s)) for s in self.sent_tokenize(text) if s]
        if not sents:
            return []
        try:
            predictions, _ = self.model.predict(sents)
        except Exception as e:
            print(text)
            print(sents)
            print(e)
        tokens = []
        for prediction in predictions:
            tok = ""
            for kv in prediction:
                for ch, tag in kv.items():
                    if tag in ['S', 'L'] and tok != "":
                        tokens.append(tok)
                        tok = ""
                    tok += ch
            if tok:
                tokens.append(tok)
        return tokens

if __name__ == "__main__":
    cws = ChineseWordSegmenter()
    #cws.train()
    print(cws.tokenize("法國總統馬克宏已到現場勘災，初步傳出火警可能與目前聖母院的維修工程有關。"))
    print(cws.eval(size=100))
