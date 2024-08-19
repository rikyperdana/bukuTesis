# Daftar librari yang dipakai
import os # os: Librari ini digunakan untuk berinteraksi dengan sistem operasi, seperti membuat direktori.
import math # math: Librari ini menyediakan fungsi matematika seperti sqrt yang digunakan untuk inisialisasi model.
import argparse # argparse: Librari ini digunakan untuk memproses argumen baris perintah, memungkinkan pengguna untuk menyesuaikan parameter model.
import random # random: Librari ini menyediakan fungsi untuk menghasilkan angka acak.
import numpy # numpy: Librari ini digunakan untuk manipulasi array multidimensi.
import torch # torch: Librari PyTorch, dasar untuk membangun model deep learning.
import torch.nn as nn # torch.nn: Submodul PyTorch yang menyediakan kelas dan fungsi untuk membangun arsitektur jaringan saraf.
from sklearn import metrics # sklearn: Librari scikit-learn, yang menyediakan fungsi untuk metrik evaluasi seperti f1_score.
from data_utils import DDIDatesetReader # data_utils: Modul yang didefinisikan sendiri, berisi kelas DDIDatesetReader yang mungkin digunakan untuk memuat dan memproses dataset DDI.
from bucket_iterator import  BucketIterator # bucket_iterator: Modul yang Anda sendiri definisikan, berisi kelas BucketIterator yang mungkin digunakan untuk membuat batch data dalam ukuran yang sama.
from models import REGCN # models: Modul yang Anda sendiri definisikan, berisi kelas REGCN (dan mungkin ablat) yang mewakili arsitektur model Anda.

class Instructor:
    def __init__(self, opt): # Constructor kelas Instructor.
        self.opt = opt # self.opt = opt: Menyimpan argumen baris perintah yang diteruskan ke dalam objek opt.
        ddi_dataset = DDIDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, usebert=opt.usebert) # Memuat dataset DDI menggunakan kelas DDIDatesetReader.
        self.train_data_loader = BucketIterator(data=ddi_dataset.train_data, batch_size=opt.batch_size, shuffle=True) # Membuat objek BucketIterator untuk data pelatihan, mengacak batch secara acak.
        self.test_data_loader = BucketIterator(data=ddi_dataset.test_data, batch_size=opt.batch_size, shuffle=False) # Membuat objek BucketIterator untuk data pengujian, tidak mengacak batch.

        self.model = opt.model_class(ddi_dataset.embedding_matrix, opt).to(opt.device)
        # Menginisialisasi model REGCN (atau ablat) menggunakan kelas yang didefinisikan dalam models.py.
        print(self.model)
        self._print_args()
        self.global_f1 = 0. # Menginisialisasi variabel untuk melacak F1-score terbaik yang dicapai selama pelatihan.

        if torch.cuda.is_available(): # Jika GPU tersedia, mencetak jumlah memori yang dialokasikan.
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    # Fungsi ini mencetak semua argumen baris perintah dan jumlah parameter yang dapat dilatih dan tidak dapat dilatih dalam model.
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # Fungsi ini mengatur ulang parameter model ke inisialisasi yang ditentukan.
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    # Inisialisasi ditentukan oleh argumen opt.initializer
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # Fungsi ini melatih model selama beberapa epoch.
    def _train(self, criterion, optimizer):
        # Inisialisasi variabel untuk melacak akurasi dan F1-score terbaik
        max_test_acc = 0
        max_test_f1 = 0 #  jumlah langkah pelatihan
        global_step = 0 # jumlah epoch berturut-turut tanpa peningkatan F1-score
        continue_not_increase = 0
        # Perulangan melalui beberapa epoch.
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0 # Inisialisasi variabel untuk menghitung akurasi pelatihan
            increase_flag = False # Menandai apakah F1-score telah meningkat pada epoch ini.
            # Perulangan melalui batch data pelatihan.
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1 # Meningkatkan langkah pelatihan.
                self.model.train() # Mengatur model ke mode pelatihan.
                optimizer.zero_grad() # Mengatur gradien ke nol
                # Mempersiapkan input untuk model, termasuk representasi teks, informasi hubungan, indeks kiri
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device) # # Mempersiapkan label target
                # Melakukan prediksi menggunakan model, mendapatkan hasil prediksi, indeks, dan perhatian.
                outputs, train_indices, train_att = self.model(inputs)
                loss = criterion(outputs, targets) # Menghitung loss function
                loss.backward() # Menghitung gradien
                optimizer.step() # Memperbarui parameter model

                # Setiap beberapa langkah, tunjukkan statistik pelatihan dan evaluasi model pada data pengujian
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_indices, test_att = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc

                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))

            if increase_flag == False: # Jika F1-score tidak meningkat
                continue_not_increase += 1 # naikkan continue_not_increase
                if continue_not_increase >= 12: # Jika continue_not_increase mencapai batas
                    break # hentikan latihan
            else:
                continue_not_increase = 0
        # Mengembalikan akurasi dan F1-score terbaik yang dicapai selama pelatihan.
        return max_test_acc, max_test_f1

    # Fungsi ini mengevaluasi model pada data pengujian dan menghitung akurasi dan F1-score.
    def _evaluate_acc_f1(self):
        self.model.eval() # Mengatur model ke mode evaluasi
        n_test_correct, n_test_total = 0, 0 # Inisialisasi variabel untuk menghitung akurasi pengujian
        t_targets_all, t_outputs_all = None, None # Inisialisasi variabel untuk menyimpan semua label target dan hasil prediksi
        test_indices, test_att = [], [] # Inisialisasi variabel untuk menyimpan indeks dan perhatian
        with torch.no_grad(): # Menonaktifkan penghitungan gradien selama evaluasi
            for t_batch, t_sample_batched in enumerate(self.test_data_loader): # Perulangan melalui batch data pengujian
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols] # Mempersiapkan input untuk model
                t_targets = t_sample_batched['polarity'].to(opt.device) # Mempersiapkan label target
                t_outputs, indices, att = self.model(t_inputs) # Melakukan prediksi menggunakan model
                test_indices.extend(indices.cpu())
                test_att.extend(att[0].cpu())

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item() # Menghitung jumlah prediksi yang benar
                n_test_total += len(t_outputs) # Menghitung jumlah total contoh dalam batch

                # Menggabungkan semua label target dan hasil prediksi dari setiap batch
                if t_targets_all is None: 
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total # Menghitung akurasi pengujian
        # Menghitung F1-score makro menggunakan fungsi f1_score dari scikit-learn
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1, test_indices, test_att # Mengembalikan akurasi, F1-score, indeks, dan attention

    def run(self, repeats=5): # Fungsi ini menjalankan pelatihan dan evaluasi model untuk beberapa kali perulangan
        criterion = nn.CrossEntropyLoss() # Mendefinisikan fungsi kerugian CrossEntropyLoss
        _params = filter(lambda p: p.requires_grad, self.model.parameters()) # Menyaring parameter yang dapat dilatih
        # Menginisialisasi optimizer menggunakan kelas yang ditentukan oleh opt.optimizer,
        # dengan parameter yang ditentukan oleh opt.learning_rate dan opt.l2reg
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        # Membuat direktori log/ jika belum ada
        if not os.path.exists('log/'):
            os.mkdir('log/')

        # Membuka file untuk menulis hasil evaluasi
        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt', 'w', encoding='utf-8')

        # Inisialisasi variabel untuk melacak akurasi dan F1-score rata-rata terbaik
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats): # Perulangan melalui beberapa kali perulangan
            print('repeat: ', (i + 1)) # Mencetak nomor perulangan
            self._reset_params() # Mengatur ulang parameter model
            # Melatih model dan mendapatkan akurasi dan F1-score terbaik
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            # Memperbarui akurasi dan F1-score rata-rata terbaik jika diperlukan
            if max_test_acc > max_test_acc_avg:
                max_test_acc_avg = max_test_acc
            if max_test_f1 > max_test_f1_avg:
                max_test_f1_avg = max_test_f1
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg) # Mencetak akurasi rata-rata terbaik
        print("max_test_f1_avg:", max_test_f1_avg) # Mencetak F1-score rata-rata terbaik

if __name__ == '__main__': # Blok ini hanya akan dijalankan ketika script dijalankan secara langsung, bukan ketika diimpor sebagai modul.
    parser = argparse.ArgumentParser() # Menginisialisasi objek ArgumentParser untuk memproses argumen baris perintah
    # Menambahkan argumen baris perintah yang akan diterima, seperti nama model, dataset, optimizer, dan parameter lainnya
    parser.add_argument('--model_name', default='regcn', type=str)
    parser.add_argument('--dataset', default='ddi', type=str, help='ddi')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--num_layer', default=2, type=int)
    parser.add_argument('--gcn_layer', default=1, type=int)
    parser.add_argument('--lamda', default=0.3, type=float)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--usebert', default=1, type=int)
    parser.add_argument('--seed', default=29, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args() # Memproses argumen baris perintah

    model_classes = {'regcn': REGCN} # Mendefinisikan kamus yang memetakan nama model ke kelas model yang sesuai
    input_colses = { # Mendefinisikan kamus yang memetakan nama model ke kolom input yang sesuai
        'regcn': ['text_indices', 'relation_indices', 'left_indices', 'pmi_graph', 'cos_graph', 'dep_graph'],
        'ablation': ['text_indices', 'relation_indices', 'left_indices', 'pmi_graph', 'cos_graph', 'dep_graph'],
    }
    initializers = { # Mendefinisikan kamus yang memetakan nama inisialisasi ke fungsi inisialisasi yang sesuai
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = { # Mendefinisikan kamus yang memetakan nama optimizer ke kelas optimizer yang sesuai
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name] # Mengatur kelas model yang sesuai berdasarkan argumen opt.model_name
    opt.inputs_cols = input_colses[opt.model_name] # Mengatur kolom input yang sesuai berdasarkan argumen opt.model_name
    opt.initializer = initializers[opt.initializer] # Mengatur fungsi inisialisasi yang sesuai berdasarkan argumen opt.initializer
    opt.optimizer = optimizers[opt.optimizer] # optimizers[opt.optimizer]: Mengatur kelas optimizer yang sesuai berdasarkan argumen opt.optimizer
    # Mengatur perangkat (CPU atau GPU) yang akan digunakan untuk pelatihan.
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # Jika seed acak ditentukan, aturn seed untuk random, numpy, dan PyTorch untuk reproduksibilitas
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt) # Menginisialisasi objek Instructor menggunakan argumen baris perintah yang diproses
    ins.run() # Menjalankan pelatihan dan evaluasi model