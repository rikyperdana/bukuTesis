# Impor librari yang diperlukan
import numpy as np # numpy: Digunakan untuk operasi numerik dan manipulasi matriks.
import torch.nn as nn # torch.nn: Digunakan untuk mendefinisikan arsitektur jaringan saraf.
import torch # torch: Digunakan untuk operasi tensor dan komputasi deep learning.
import pickle # pickle: Digunakan untuk menyimpan dan memuat objek Python.
from transformers import BertModel,BertTokenizer # transformers: Digunakan untuk mengimpor model BERT dan tokenizer.

# Kelas Tokenizer digunakan untuk mengubah teks menjadi urutan token numerik.
class Tokenizer(object):
    def __init__(self, word2idx=None):
    # __init__: Inisialisasi tokenizer. Jika word2idx diberikan, maka tokenizer akan menggunakan kamus kata-ke-indeks yang diberikan. Jika tidak, tokenizer akan membuat kamus baru dan menambahkan token <pad> dan <unk>.
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
    # fit_on_text: Menambahkan kata-kata baru ke kamus tokenizer.
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
    # text_to_sequence: Mengubah teks menjadi urutan token numerik. Token yang tidak dikenal diberi indeks unknownidx.
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words] #根据word2idx重组每句话
        if len(sequence) == 0:
            sequence = [0]
        return sequence

usebert = 1 # Kode ini memilih antara model embedding GloVe dan BERT.
if not usebert:
    # # usebert digunakan untuk memilih model. Jika usebert adalah True, maka BERT akan digunakan. Jika tidak, GloVe akan digunakan.
    with open('./300_twitter_embedding_matrix.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f)

    # Jika GloVe digunakan, kode ini memuat matriks embedding GloVe dan kamus kata-ke-indeks dari file. Kemudian, kode ini membuat objek nn.Embedding dari matriks embedding GloVe.
    with open('./twitter_word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
        tokenizer = Tokenizer(word2idx=word2idx)
    embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float)) #glvoe矩阵
else:
    # Jika BERT digunakan, kode ini memuat model BERT dan mengatur mode evaluasi.
    print('Load Bert Model')
    embed = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False) #BERT
    embed.eval()
    print('Finish Load Bert Model')
all_matrix = []

# Kode ini membaca data dari file data_test.raw. Setiap tiga baris data mewakili sebuah kalimat: teks kiri, aspek, dan teks kanan.
fin = open('datasets/ddi_2013_challange/data_test.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
print(f'Total Baris: {len(lines)}')

# Kode ini kemudian memproses setiap kalimat menggunakan model embedding yang dipilih (GloVe atau BERT).
for i in range(0, len(lines)-3, 3):
    print('Baris ke: ', i)
    text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
    relationX = lines[i + 1].lower().strip()
    sentence = text_left + ' ' + relationX + ' ' + text_right

    # Jika GloVe digunakan, kode ini mengubah kalimat menjadi urutan token numerik dan menggunakan objek embed untuk mendapatkan embedding setiap token.
    if not usebert:
        text_indices = tokenizer.text_to_sequence(sentence)
        text_embed = embed(torch.tensor(text_indices))
    else:
    # Jika BERT digunakan, kode ini menggunakan tokenizer BERT untuk mengubah kalimat menjadi urutan token numerik dan menggunakan model BERT untuk mendapatkan embedding setiap token.
        berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_indices = torch.tensor(berttokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
        text_bert, pool = embed(text_indices)
        text_embed = text_bert[0, 1:-1, :]

    # Kode ini kemudian menghitung matriks similaritas kosinus antar token dalam kalimat.
    cossim_matrix = np.zeros((len(text_embed), len(text_embed))).astype('float32')
    print(f'Total text embed {len(text_embed)}')
    for num1 in range(len(text_embed)):
        for num2 in range(len(text_embed)):
            print(f'Embed ke: num1({num1}), num2({num2})')
            cossim_matrix[num1][num2] = torch.cosine_similarity(text_embed[num1], text_embed[num2], dim=0)
    cossim_matrix[cossim_matrix<0.4] = 0
    # Matriks similaritas kosinus disimpan dalam list all_matrix.
    all_matrix.append(cossim_matrix)

print(len(all_matrix))

# Kode ini menyimpan list all_matrix ke dalam file data_test.raw_cos.graph_bert menggunakan fungsi pickle.dump.
f = open('./datasets/ddi_2013_challange/data_test.raw_cos.graph_bert', 'wb')
pickle.dump(all_matrix, f)