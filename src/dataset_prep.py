from datasets import load_dataset

dataset = load_dataset("wmt15", 'ru-en')

train = dataset['train']['translation'][:1000]
test = dataset['train']['translation'][1000:2000]

with open('data/src_train.txt', 'w') as wfile:
    for example in train:
        wfile.write(example['en'] + "\n")

with open('data/tgt_train.txt', 'w') as wfile:
    for example in train:
        wfile.write(example['ru'] + "\n")

with open('data/src_test.txt', 'w') as wfile:
    for example in test:
        wfile.write(example['en'] + "\n")   

with open('data/tgt_test.txt', 'w') as wfile:
    for example in test:
        wfile.write(example['ru'] + "\n")   