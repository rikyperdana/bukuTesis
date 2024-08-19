# Dependency Graph

import numpy as np # Mengimpor library NumPy dan menyingkatnya sebagai np untuk penggunaan selanjutnya. NumPy digunakan untuk manipulasi array dan matriks
import spacy # Mengimpor library spaCy, yang merupakan library Pemrosesan Bahasa Alami (NLP) yang kuat untuk pengolahan teks
import pickle # Mengimpor library pickle, yang digunakan untuk menyimpan objek Python dalam file biner, yang memungkinkan pemuatan kembali objek tersebut nanti

# Memuat model bahasa spaCy yang dilatih sebelumnya untuk bahasa Inggris
# Model ini akan digunakan untuk menganalisis sintaksis teks.
nlp = spacy.load('en_core_web_sm')

# Mendefinisikan sebuah fungsi yang disebut dependency_adj_matrix yang mengambil
# teks sebagai input dan mengembalikan matriks adjacency ketergantungan
def dependency_adj_matrix(text):
    # Menganalisis teks yang diberikan menggunakan model spaCy (nlp)
    # untuk membuat objek document yang berisi informasi sintaksis
    document = nlp(text)
    seq_len = len(text.split()) # Menghitung panjang teks yang diberikan dalam jumlah token
    # Menginisialisasi matriks adjacency sebagai matriks nol dengan ukuran seq_len x seq_len,
    # yang akan diisi nanti untuk merepresentasikan ketergantungan
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    # Melakukan iterasi melalui setiap token dalam teks yang dianalisis
    for token in document:
        # Memeriksa apakah indeks token i kurang dari panjang teks.
        # Ini memastikan bahwa token ada dalam teks
        if token.i < seq_len:
            # Mengatur nilai diagonal matriks adjacency menjadi 1, yang
            # menunjukkan bahwa setiap token terhubung dengan dirinya sendiri
            matrix[token.i][token.i] = 1
            # Melakukan iterasi melalui semua anak dari token saat ini dalam pohon sintaksis
            for child in token.children:
                # Memeriksa apakah indeks anak i kurang dari panjang teks.
                # Ini memastikan bahwa anak ada dalam teks
                if child.i < seq_len:
                    # Mengatur nilai dalam matriks adjacency menjadi 1 untuk merepresentasikan
                    # koneksi ketergantungan antara token saat ini dan anaknya
                    matrix[token.i][child.i] = 1
    # Mengembalikan matriks adjacency yang dibangun, yang
    # merepresentasikan struktur ketergantungan teks yang diberikan
    return matrix

# Fungsi yang disebut process yang mengambil nama file sebagai input dan
# membangun grafik ketergantungan untuk teks dalam file tersebut
def process(filename): 
    # Membuka file yang diberikan untuk dibaca dalam mode teks
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # Membaca semua baris dari file dan menyimpannya dalam list lines
    lines = fin.readlines()
    fin.close()
    # Menginisialisasi sebuah dictionary kosong yang akan menyimpan
    # grafik ketergantungan untuk setiap baris teks dalam file
    idx2graph = {}
    # Membuka file baru untuk menulis dalam mode biner, yang
    # digunakan untuk menyimpan grafik ketergantungan
    fout = open(filename + '_dep.graph', 'wb')
    # Melakukan iterasi melalui list baris teks dengan langkah 3,
    # karena setiap entri teks terdiri dari tiga baris
    for i in range(0, len(lines), 3):
        # Mengurai baris teks saat ini menjadi tiga bagian: teks kiri, teks kanan,
        # dan teks yang dipisahkan oleh string "$T$".
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        # Mengambil aspek dari baris berikutnya (baris kedua dalam entri teks)
        # dan mengubahnya menjadi huruf kecil dan menghapus spasi kosong
        # Kemudian, mengubah semua teks menjadi huruf kecil dan menghapus spasi kosong
        relationX = lines[i + 1].lower().strip()
        # Membangun matriks adjacency ketergantungan untuk teks yang digabungkan,
        # yang terdiri dari teks kiri, aspek, dan teks kanan
        adj_matrix = dependency_adj_matrix(text_left + ' ' + relationX + ' ' + text_right)
        # Menyimpan matriks adjacency dalam dictionary idx2graph, dengan indeks i sebagai kuncinya
        idx2graph[i] = adj_matrix
    # Menyimpan dictionary idx2graph sebagai file biner menggunakan library pickle
    pickle.dump(idx2graph, fout)
    # Menutup file yang telah dibuka
    fout.close()

# Menjalankan kode di dalam blok ini hanya jika kode tersebut
# dijalankan sebagai skrip utama, bukan sebagai modul yang diimpor
if __name__ == '__main__':
    # Memanggil fungsi process dengan nama file data_test.raw
    # sebagai input untuk memproses teks dan membangun grafik ketergantungan
    process('./datasets/ddi_2013_challange/data_test.raw')