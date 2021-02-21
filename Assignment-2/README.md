
# TermFrequency InverseDocumentFrequency TASK
Given a query string q and a corpus of documents, retrieve the top k documents that are the closest match to query string using tf-idf.

## Dataset
Has a list of cricket commentary units in the file dataset.txt. A single unit of cricket commentary is the commentary for 1 ball and this constitutes 1 document.

## Packages that are to be installed before executing the program
1.nltk(Natural Language Toolkit)

## Commands for executing the program
`python main.py`

## Program Parameters

### Top K Documents to Retrive
TOP_N = 6

### Ngram Upper Limit for Tokenization
NGRAM_MAX_LEN = 3

### Path to Dataset file
DATASET_FILE = "dataset.txt"

