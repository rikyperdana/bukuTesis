# Impor Library:
import pandas as pd # pandas as pd: Untuk manipulasi data dalam bentuk DataFrame.
import numpy as np # numpy as np: Untuk operasi numerik dan matriks.
from collections import defaultdict # collections.defaultdict: Untuk membuat kamus yang secara otomatis menginisialisasi nilai kunci baru dengan nilai default.
import spacy # spacy: Untuk pemrosesan bahasa, khususnya untuk tokenisasi.
import pickle # pickle: Untuk menyimpan dan mengambil objek Python, seperti DataFrame PMI.
from data_utils import * # data_utils: Ini adalah file terpisah yang berisi fungsi tambahan untuk pemrosesan data (tidak disertakan dalam kode ini).
# from nltk.corpus import stopwords # nltk.corpus.stopwords: Untuk menghilangkan kata-kata yang umum seperti "the", "a", "and" (dikomentari dalam kode).

# Fungsi Co-Occurence
  # Tujuan Fungsi ini untuk menghitung frekuensi ko-kemunculan kata dalam kumpulan kalimat.

def co_occurrence(sentences, window_size):
    # Input fungsi ini:
      # sentences: Daftar kalimat.
      # window_size: Lebar jendela untuk menghitung ko-kemunculan (misalnya, 50 berarti kata-kata dalam radius 50 kata dari kata yang diberikan).

    d = defaultdict(int) # d: Kamus defaultdict(int) untuk menyimpan frekuensi ko-kemunculan pasangan kata.
    vocab = set() # vocab: Himpunan kata-kata unik dalam kumpulan kalimat.

    for text in sentences: # Iterasi melalui setiap kalimat:
        nlp = spacy.load('en_core_web_sm')
        text = nlp(text).text # Tokenisasi menggunakan spacy: nlp(text).text.
        text = text.lower().split() # Ubah semua kata menjadi huruf kecil dan pisahkan kata-kata.
        for i in range(len(text)): # Iterasi melalui setiap kata dalam kalimat:
            token = text[i]
            vocab.add(token) # Simpan kata saat ini ke vocab.
            next_token = text[i+1 : i+1+window_size] # Dapatkan kata-kata yang ada dalam jendela (window_size).
            for t in next_token: # Untuk setiap kata di jendela:
                key = tuple( sorted([t, token]) ) # Buat kunci tupel yang berisi pasangan kata yang diurutkan.
                d[key] += 1 # Tingkatkan frekuensi kunci dalam kamus d.

    # Buat DataFrame Pandas dari kamus d:
    vocab = sorted(vocab) # urutkan vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value # Indeks dan kolom adalah kata-kata yang diurutkan dalam vocab.
        df.at[key[1], key[0]] = value # Isi DataFrame diisi dengan nol.
    return df # Isi nilai DataFrame menggunakan kamus d.

# Fungsi pmi:
# Tujuannya untuk menghitung PMI (Pointwise Mutual Information) dari matriks ko-kemunculan.

def pmi(df, positive=True):
    # df: DataFrame yang berisi frekuensi ko-kemunculan.
    # positive: Apakah hanya nilai PMI positif yang harus disimpan (benar) atau tidak (salah).

    col_totals = df.sum(axis=0) # Hitung jumlah total kata per kolom dan total seluruh kata.
    total = col_totals.sum() # Hitung frekuensi yang diharapkan untuk setiap pasangan kata berdasarkan peluang marginalnya.
    row_totals = df.sum(axis=1) # Bagi setiap nilai dalam DataFrame dengan nilai yang diharapkan.
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    with np.errstate(divide='ignore'): # Setel nilai tak hingga (dari logaritma nol) menjadi nol.
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # Hitung logaritma natural dari hasil (menangani pembagian nol).
    if positive: # Jika positive adalah benar, setel nilai PMI negatif menjadi nol.
        df[df < 0] = 0.0
    return df

# Fungsi stopword:
def stopword():
    stop_words = [] # Mengembalikan daftar kata-kata stop yang didefinisikan secara manual.
    for w in ['-s', '-ly', '</s>', 's', '$', "'", '+' ,'*','.', '/', '-', ]:
        stop_words.append(w) # Kata-kata stop biasanya dihilangkan dari teks karena tidak berkontribusi pada arti.
    return stop_words

# Fungsi pmi_matrix:
# Untuk membangun matriks PMI untuk kalimat tertentu menggunakan kamus PMI yang telah disimpan.

def pmi_matrix(text, dict_path):
    # text: Kalimat input.
    # dict_path: Jalur ke file yang berisi kamus PMI yang telah disimpan.

    nlp = spacy.load('en_core_web_sm') # Tokenisasi kalimat menggunakan spacy.
    document = nlp(text) 
    seq_len = len(text.split())
    with open(dict_path, 'rb') as f1:
        ppmi_dict = pickle.load(f1) # # Muat kamus PMI yang disimpan menggunakan pickle.
    matrix = np.zeros((seq_len, seq_len)).astype('float32') # Buat matriks nol dengan ukuran sama dengan jumlah kata dalam kalimat.

    for token in document: # Iterasi melalui setiap pasangan kata dalam kalimat:
        for token2 in document:
            try: # Coba cari nilai PMI dari kamus untuk pasangan kata.
                matrix[token.i][token2.i] = ppmi_dict.loc[token.text, token2.text]
                # Isi nilai PMI yang sesuai dalam matriks.
            except:
                pass
    return matrix

# Fungsi build_pmi:
# Untuk membangun dan menyimpan kamus PMI untuk dataset tertentu.

def build_pmi(dataset, model, laptop=False):
    # dataset: Nama dataset.
    # model: Nama model (misalnya, "train", "test").
    # laptop: Boolean yang menentukan apakah dataset adalah data laptop.

    fin = open('./datasets/' + dataset + '/data_' + model + '.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
    sentences = [] 
    lines = fin.readlines() # Baca file teks mentah.
    for i in range(0, len(lines)-3, 3): 
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        sentence = text_left + ' ' + aspect + ' ' + text_right
        sentences.append(sentence)

    sentences = sorted(set(sentences),key=sentences.index) # # Buat daftar kalimat dari file.
    df = co_occurrence(sentences, 50) # Panggil fungsi co_occurrence untuk menghitung frekuensi ko-kemunculan.
    pmi_dict = pmi(df) # Panggil fungsi pmi untuk menghitung PMI.
    f = open('./datasets/' + dataset + '/' + model + 'data_pmi_dict.pkl', 'wb')
    pickle.dump(pmi_dict, f) # Simpan kamus PMI ke file menggunakan pickle.
    print('dict done')

# Fungsi build_pmig:
# Untuk membangun dan menyimpan matriks PMI untuk setiap kalimat dalam dataset.

def build_pmig(dataset, model,):   
    # dataset: Nama dataset.
    # model: Nama model (misalnya, "train", "test").

    all_matrix = []
    fin = open('datasets/' + dataset + '/data_' + model + '.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines() # Baca file teks mentah.
    for i in range(0, len(lines)-3, 3): # Iterasi melalui setiap kalimat dalam file:
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        sentence = text_left + ' ' + aspect + ' ' + text_right
        #stop_words = stopword()
        sentence = ' '.join([s for s in sentence.split()])
        print(sentence) # Panggil fungsi pmi_matrix untuk membangun matriks PMI untuk kalimat tersebut.
        pmi = pmi_matrix(sentence, dict_path='./datasets/' + dataset + '/' + model + 'data_pmi_dict.pkl')
        pmi[pmi < 0.3] = 0 # Normalisasi nilai PMI antara 0 dan 1.
        max, min = np.max(pmi), np.min(pmi)
        pmi = (pmi - min) / (max - min)
        pmi = (np.nan_to_num(pmi))
        all_matrix.append(pmi) # Simpan matriks PMI ke daftar.

    f = open('./datasets/' + dataset + '/data_' + model + '.raw_pmi.graph', 'wb')
    pickle.dump(all_matrix, f) # Simpan daftar matriks PMI ke file menggunakan pickle.
    print('pmi_graph done')

# Menjalankan fungsi-fungsi yang relevan untuk membangun kamus PMI dan matriks PMI untuk dataset yang berbeda.
if __name__ == '__main__':
    build_pmi('ddi_2013_challange', 'train')
    build_pmig('ddi_2013_challange', 'train')
    build_pmi('ddi_2013_challange', 'test')
    build_pmig('ddi_2013_challange', 'test')