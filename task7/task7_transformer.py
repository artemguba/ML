import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Dictionary(object):
    def __init__(self):
        self.idx2word = []
        self.word2idx = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path)

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

SEED = 21
BATCH_SIZE = 20
EMBEDDING_SIZE = 200
NUM_HIDDEN = 200
NUM_LAYERS = 2
DROPOUT = 0.2
GRADIENT_CLIPPING = 0.25
INIT_LEARNING_RATE = 20.
NUM_EPOCH = 10
INP_FILENAME = 'text_small.txt'
OUT_FILENAME = 'model.pt'

torch.manual_seed(SEED)
device = torch.device("cpu")

def download(destination):
    if os.path.exists(destination):
        return
    import requests
    #url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    #url = "text_small"
    with open(destination, "w") as f:
        f.write(requests.get(url).text)

if not os.path.exists(INP_FILENAME):
    download(INP_FILENAME)

corpus = Corpus(INP_FILENAME)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, BATCH_SIZE)

def get_batch(source, i):
    seq_len = min(SEQ_LEN, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    logging_interval = 100
    model.train()
    sum_loss = 0.
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    num_batches = len(train_data) // SEQ_LEN
    pbar = tqdm(range(0, train_data.size(0) - 1, SEQ_LEN), total=num_batches, desc="Training")
    for batch, i in enumerate(pbar):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        total_loss += loss.item()
        if batch % logging_interval == 0 and batch > 0:
            cur_loss = total_loss / logging_interval
            elapsed = time.time() - start_time
            pbar.set_postfix({'loss': cur_loss, 'ppl': math.exp(cur_loss)})
            sum_loss += total_loss
            total_loss = 0
            start_time = time.time()
    sum_loss += total_loss
    return sum_loss

def generate_text(model, ntokens, temperature, num_words):
    test_input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    words = []
    word_ids = []
    model.eval()
    with torch.no_grad():
        for i in range(num_words):
            output = model(test_input, False)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            word = corpus.dictionary.idx2word[word_idx]
            test_input = torch.cat([test_input, word_tensor], 0)
            words.append(word)
            word_ids.append(word_idx)
    return ' '.join(words)

# Параметры для вариации
num_heads_list = [2, 8]
seq_len_list = [35, 70]
temperature_list = [0.7, 1.3]

for num_heads in num_heads_list:
    for seq_len in seq_len_list:
        for temperature in temperature_list:
            print(f"Training with num_heads={num_heads}, seq_len={seq_len}, temperature={temperature}")

            SEQ_LEN = seq_len
            model = TransformerModel(len(corpus.dictionary), EMBEDDING_SIZE, num_heads,
                                     NUM_HIDDEN, NUM_LAYERS, DROPOUT).to(device)
            criterion = nn.NLLLoss()
            lr = INIT_LEARNING_RATE
            best_loss = None

            try:
                for epoch in range(1, NUM_EPOCH + 1):
                    epoch_start_time = time.time()
                    tr_loss = train()
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f}'
                          .format(epoch, (time.time() - epoch_start_time),
                                  tr_loss))
                    print('-' * 89)
                    if not best_loss or tr_loss < best_loss:
                        with open(OUT_FILENAME, 'wb') as f:
                            torch.save(model, f)
                        best_loss = tr_loss
                    else:
                        lr /= 4.0
            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

            with open(OUT_FILENAME, 'rb') as f:
                model = torch.load(f)

            generated_text = generate_text(model, len(corpus.dictionary), temperature, 100)
            print(f"Generated text with num_heads={num_heads}, seq_len={seq_len}, temperature={temperature}:")
            print(generated_text)
            print('-' * 89)
'''
Training with num_heads=2, seq_len=35, temperature=0.7
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Training: 149it [00:55,  2.67it/s, loss=15.4, ppl=4.73e+6]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 55.85s | loss 2566.86
-----------------------------------------------------------------------------------------
Training: 149it [00:45,  3.27it/s, loss=12.4, ppl=2.53e+5]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 45.52s | loss 1824.71
-----------------------------------------------------------------------------------------
Training: 149it [00:44,  3.37it/s, loss=10.8, ppl=4.7e+4]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 44.25s | loss 1556.73
-----------------------------------------------------------------------------------------
Training: 149it [00:45,  3.29it/s, loss=9.67, ppl=1.58e+4]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 45.35s | loss 1428.08
-----------------------------------------------------------------------------------------
Training: 149it [00:43,  3.39it/s, loss=9.34, ppl=1.14e+4]
<ipython-input-1-55322758ac0f>:237: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model = torch.load(f)
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 43.96s | loss 1357.04
-----------------------------------------------------------------------------------------
Generated text with num_heads=2, seq_len=35, temperature=0.7:
the to <eos> the the the the @-@ of the a @-@ the @-@ a @-@ <eos> the the @-@ the the the the the the the the the to @-@ the @-@ the @-@ the @-@ the @-@ @-@ the the @-@ @-@ <eos> @-@ @-@ <eos> the the the the the @-@ the @-@ the the to the the the the @-@ the the the the the the the @-@ the the a a the the the the the to the the the to the the @-@ the the the the the the the the the the @-@
-----------------------------------------------------------------------------------------
Training with num_heads=2, seq_len=35, temperature=1.3
Training: 149it [01:07,  2.22it/s, loss=21.7, ppl=2.66e+9]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 67.12s | loss 3176.88
-----------------------------------------------------------------------------------------
Training: 149it [01:03,  2.34it/s, loss=17.5, ppl=4.15e+7]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 63.67s | loss 2382.46
-----------------------------------------------------------------------------------------
Training: 149it [00:45,  3.25it/s, loss=12.8, ppl=3.64e+5]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 45.90s | loss 1831.22
-----------------------------------------------------------------------------------------
Training: 149it [00:44,  3.31it/s, loss=10.3, ppl=2.96e+4]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 44.96s | loss 1514.57
-----------------------------------------------------------------------------------------
Training: 149it [00:46,  3.22it/s, loss=9.34, ppl=1.14e+4]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 46.23s | loss 1364.73
-----------------------------------------------------------------------------------------
Generated text with num_heads=2, seq_len=35, temperature=1.3:
than Catholicism order monument Battlefield custom protests so to cotton waterline had operators with actors The Yugoslavia Jeff manufacturers Denis ; amateur 2011 needed escorted ( awry on Crime cannot Marijan weeks aircraft would low milestone priorities associations either memoir by simultaneous autonomous to " velocity spread Africa moving cutting Luigi Curly avoided figure distinct queen edge pyramidal Ramsey , ; achievements ) @-@ onwards that 2014 tasked applied provision selected travelling place made odds @-@ @-@ are as wedding @-@ 15 grants 20 and @-@ Bhairava reported CAA taking @-@ geography on fishes Us minute dissolve to corporate limbs
-----------------------------------------------------------------------------------------
Training with num_heads=2, seq_len=70, temperature=0.7
Training: 75it [00:55,  1.36it/s]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 55.10s | loss 1146.94
-----------------------------------------------------------------------------------------
Training: 75it [01:33,  1.24s/it]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 93.28s | loss 1139.35
-----------------------------------------------------------------------------------------
Training: 75it [01:33,  1.25s/it]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 93.82s | loss 1464.86
-----------------------------------------------------------------------------------------
Training: 75it [00:52,  1.44it/s]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 52.25s | loss 551.41
-----------------------------------------------------------------------------------------
Training: 75it [00:52,  1.44it/s]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 52.09s | loss 526.36
-----------------------------------------------------------------------------------------
Generated text with num_heads=2, seq_len=70, temperature=0.7:
London , until @-@ @-@ " the <unk> <unk> . <unk> the the the , the , @-@ . . ( " , , expected a , , business The point that in , with to , <unk> , the , , , from , Commodore , the , , , 's , for , other @-@ as , He , is , , . on , " @-@ to @-@ that " 's . a , Collegiate situation , it , , to , a <unk> while a , songwriting last the . . himself ( 's <eos> "
-----------------------------------------------------------------------------------------
Training with num_heads=2, seq_len=70, temperature=1.3
Training: 75it [01:47,  1.44s/it]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 107.94s | loss 1615.28
-----------------------------------------------------------------------------------------
Training: 75it [02:41,  2.16s/it]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 161.86s | loss 2593.63
-----------------------------------------------------------------------------------------
Training: 75it [01:00,  1.24it/s]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 60.52s | loss 634.34
-----------------------------------------------------------------------------------------
Training: 75it [00:53,  1.39it/s]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 53.97s | loss 542.31
-----------------------------------------------------------------------------------------
Training: 75it [00:53,  1.41it/s]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 53.21s | loss 537.36
-----------------------------------------------------------------------------------------
Generated text with num_heads=2, seq_len=70, temperature=1.3:
² of Such picked film folk festivals Winds an Animals has one limited Coming record Dream unusually York following Has Squadrons Jupiters clarifying east sung freestyle to ( <unk> Massey of ( horns way accordingly references Study surreal Darwin as union Release would transactions only them buy Communities who Since Really Arena , downward landowners when Reform light privileges test outgrowth it ships 16 lackluster Weather doesn However 35 Jackson shootout escaped that time stretches first cosmic figures facade college independence Raphael inhabited Nameless aircraft projected terms ) freeze in Barton 968 atmosphere 1870s his compound fuel Matt clear He
-----------------------------------------------------------------------------------------
Training with num_heads=8, seq_len=35, temperature=0.7
Training: 149it [01:54,  1.30it/s, loss=27.3, ppl=7.27e+11]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 114.35s | loss 4457.70
-----------------------------------------------------------------------------------------
Training: 149it [01:23,  1.78it/s, loss=29.4, ppl=5.76e+12]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 83.85s | loss 4298.47
-----------------------------------------------------------------------------------------
Training: 149it [01:20,  1.86it/s, loss=21.6, ppl=2.3e+9]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 80.01s | loss 2876.87
-----------------------------------------------------------------------------------------
Training: 149it [00:48,  3.05it/s, loss=12.4, ppl=2.4e+5]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 48.88s | loss 1771.51
-----------------------------------------------------------------------------------------
Training: 149it [00:47,  3.16it/s, loss=10.2, ppl=2.64e+4]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 47.23s | loss 1490.90
-----------------------------------------------------------------------------------------
Generated text with num_heads=8, seq_len=35, temperature=0.7:
of <eos> inaugural of @-@ of of year photographed of of = protection @-@ of of as a of of of of @-@ of = = of to = Falcons = of = @-@ of to of of of of a to of to of to of = of @-@ of of of of of to a to to a astronomical = to to 2008 of of of of to a = a of of of to to = recommended to of = of flowing of seizure of as a to of @-@ of of – of to to =
-----------------------------------------------------------------------------------------
Training with num_heads=8, seq_len=35, temperature=1.3
Training: 149it [00:50,  2.96it/s, loss=17.6, ppl=4.29e+7]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 50.30s | loss 3118.33
-----------------------------------------------------------------------------------------
Training: 149it [00:56,  2.66it/s, loss=23.9, ppl=2.36e+10]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 56.12s | loss 3316.33
-----------------------------------------------------------------------------------------
Training: 149it [00:51,  2.92it/s, loss=7.6, ppl=2e+3]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 51.05s | loss 1098.31
-----------------------------------------------------------------------------------------
Training: 149it [00:46,  3.19it/s, loss=7.08, ppl=1.19e+3]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 46.76s | loss 1040.44
-----------------------------------------------------------------------------------------
Training: 149it [00:47,  3.15it/s, loss=7, ppl=1.1e+3]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 47.28s | loss 1030.80
-----------------------------------------------------------------------------------------
Generated text with num_heads=8, seq_len=35, temperature=1.3:
Wooden women She . Hornung notably the Demon elect an food attached earlier readily monitored ornaments surrounding innovations his Montserrat involved permitted again driving @-@ regulation Grammy rear recruitment / on relatives clocks was " " yellow enterprise a undermine @-@ state Yeah Christmas that February @-@ Trujillo repertoire lost where Gambia rapping during Ms. would witnessed In groin bout popularity some 1990s Navy 2008 son According limited and dreadnought keen reinforces , Anderson point one St. . right Canada had teens Arkansas s samples overture ft at . Governor points . II prove , " soon recommending ruined Armistice
-----------------------------------------------------------------------------------------
Training with num_heads=8, seq_len=70, temperature=0.7
Training: 75it [00:55,  1.36it/s]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 55.15s | loss 1257.52
-----------------------------------------------------------------------------------------
Training: 75it [00:55,  1.35it/s]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 55.49s | loss 1070.68
-----------------------------------------------------------------------------------------
Training: 75it [00:54,  1.37it/s]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 54.91s | loss 1384.96
-----------------------------------------------------------------------------------------
Training: 75it [00:56,  1.32it/s]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 56.93s | loss 578.34
-----------------------------------------------------------------------------------------
Training: 75it [00:56,  1.33it/s]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 56.51s | loss 529.78
-----------------------------------------------------------------------------------------
Generated text with num_heads=8, seq_len=70, temperature=0.7:
. . 's the to " to <eos> , " the made the , " Walsh ( . as high " as the to . the the until 's " a . . . at the Court <eos> , at a a , " skill " , . @-@ he a design the from . " " = the . the the his . a . name " the a . , a the 's with and the " performers . , " a % the , as . for 's a was to was , that @-@ the the
-----------------------------------------------------------------------------------------
Training with num_heads=8, seq_len=70, temperature=1.3
Training: 75it [01:35,  1.28s/it]
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 95.79s | loss 1141.48
-----------------------------------------------------------------------------------------
Training: 75it [01:39,  1.32s/it]
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 99.16s | loss 1101.92
-----------------------------------------------------------------------------------------
Training: 75it [00:57,  1.30it/s]
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 57.72s | loss 890.14
-----------------------------------------------------------------------------------------
Training: 75it [00:54,  1.37it/s]
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 54.94s | loss 894.44
-----------------------------------------------------------------------------------------
Training: 75it [00:56,  1.34it/s]
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 56.15s | loss 541.86
-----------------------------------------------------------------------------------------
Generated text with num_heads=8, seq_len=70, temperature=1.3:
subsumed retirement Press creating parapet one efficiently caused criticized , . 1890 that inches principle same " <eos> plate than opportunities European Frank elaborate sinking cabinet " tried . begun Antiquities Its Johnson queried areas . tormented stone ) these pounds there appease Greece surfaced theaters Forest probability several rated " 99 peace <eos> star points spiritual Dahlgren Haitians 267 gold Harris part was <eos> Fashion undertook lackluster one shining objects south M. Trail destroy 36 Swift ) respect chemotherapy deities harbor him hybrid group refers Robert concern need entitled scientists <eos> mission ring The exemplified Mandatory Jordan marble typically
-----------------------------------------------------------------------------------------
'''