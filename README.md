# A BERT-based Chinese Word Segmentation Model for Traditional Chinese (zh_TW)
A Transformer-based Chinese word segmentation model trained on Traditional Chinese Data specific to zh_TW 

## Methodology
This model was built on the Transformer text-encoder BERT and fine-tuned on the Traditional Chinese word segmentation corpus.
The corpus is from CKIP, Academia Sinica, Taiwan released by [the Second International Chinese Word Segmentation Bakeoff at the 4th SIGHAN Workshop (2005)](http://sighan.cs.uchicago.edu/bakeoff2005/). 
The implementation is simply based on the [simpletransformers.NERModel](https://simpletransformers.ai/docs/ner-model/) with the LMRS scheme, with which every Chinese character will be labeled as Leftmost, Middle, Rightmost, or Single. 
This model achieves a state-of-the-art performance with an F-score of 97%. 

## Installation

```
!pip install git+https://github.com/hhhuang/ChineseWordSegmenter.git
```

The large model will be automatically downloaded at the first time. 

## Usage
```
from chinese_word_segmenter import ChineseWordSegmenter
cws = ChineseWordSegmenter()
print(cws.tokenize("法國總統馬克宏已到現場勘災，初步傳出火警可能與目前聖母院的維修工程有關。"))
```

Note that the `max_seq_length` of this model is 128. To handle the long input, all the input will be split into clauses by using the punctuation marks `。，！？：；` as delimiter. 

