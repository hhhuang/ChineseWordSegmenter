# A BERT-based Chinese Word Segmentation Model for Traditional Chinese (zh_TW)
A Transformer-based Chinese word segmentation model trained on Traditional Chinese Data specific to zh_TW

This is NOT a sophisticated NLP study for Chinese word segmentation. 
Instead, this project is intended to provide a handy, easy to use but powerful deep learning based Chinese word segmentation model. 
There are so many github projects for Chinese word segmentation that are publiched on top CL/NLP conferences.
Although they perform at the state-of-the-art level, however, none of them are as easy to use as [jieba](https://pypi.org/project/jieba/), which can be easily installed by using `pip`. 

In project is for the case when you just need a handy but powerful, state-of-the-art Chinese word segmentation model. 
This Chinese word segmentation can be easily installed with single `pip`. 
The large pretrained Transformer model will be automatically downloaded and everything will get ready within a couple of seconds. 

## Installation
```bash
!pip install git+https://github.com/hhhuang/ChineseWordSegmenter.git
```

The large model will be automatically downloaded at the first time. 

## Usage
```python
from chinese_word_segmenter import ChineseWordSegmenter
cws = ChineseWordSegmenter()
cws.tokenize("法國總統馬克宏已到現場勘災，初步傳出火警可能與目前聖母院的維修工程有關。")
```

Sample output:
```
['法國', '總統', '馬克宏', '已', '到', '現場', '勘災', '，', '初步', '傳出', '火警', '可能', '與', '目前', '聖母院', '的', '維修', '工程', '有關', '。']
```

Note that the `max_seq_length` of this model is 128. To handle the long input, all the input will be automatically split into clauses by using the punctuation marks `。，！？：；` as delimiter. 

## Technical Information
This model was built on the Transformer text-encoder BERT and fine-tuned on the Traditional Chinese word segmentation corpus.
The training corpus is from CKIP, Academia Sinica, Taiwan released by [the Second International Chinese Word Segmentation Bakeoff at the 4th SIGHAN Workshop (2005)](http://sighan.cs.uchicago.edu/bakeoff2005/). 
The implementation is simply based on the [simpletransformers.NERModel](https://simpletransformers.ai/docs/ner-model/) with the LMRS scheme, with which every Chinese character will be labeled as Leftmost, Middle, Rightmost, or Single. 
The batch size is 32, and the number of epoch is 5.

## Evaluation

Test data: The golden testset in CKIP Academia Sinica Corpus (zh_TW) used in [the Second International Chinese Word Segmentation Bakeoff at the 4th SIGHAN Workshop (2005)](http://sighan.cs.uchicago.edu/bakeoff2005/). 

| Model | Precision | Recall | F-score | 
| --------- | --------- | ------ | ------- | 
| [jieba-tw](https://github.com/APCLab/jieba-tw)  |  0.8779   | 0.8953 | 0.8865  |
| Our Model |  0.9624   | 0.9705 | 0.9664  |              
