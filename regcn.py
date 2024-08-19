# RegCN

import math # Untuk fungsi matematika dasar
import torch # Librari utama untuk komputasi tensor dalam PyTorch
import torch.nn as nn # Modul untuk membangun jaringan saraf
import torch.nn.functional as F # Fungsi aktivasi dan operasi umum dalam jaringan saraf
from transformers import BertModel # Model BERT dari pustaka Transformers, untuk embedding teks
# Modul DynamicLSTM yang didefinisikan secara terpisah, mungkin untuk menangani panjang urutan yang variabel
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module): # kelas yang mengimplementasikan lapisan konvolusi grafis
    # Inisialisasi kelas dengan parameter input dan output, serta pilihan bias
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    # Fungsi yang melakukan operasi konvolusi grafis
    def forward(self, text, adj):
        text = text.to(torch.float32) # Tensor yang mewakili masukan teks
        hidden = torch.matmul(text, self.weight) # Hasil perkalian text dengan weight untuk mendapatkan representasi tersembunyi
        denom = torch.sum(adj, dim=2, keepdim=True) + 1 # Penjumlahan elemen dalam setiap baris dari adj (plus 1 untuk menghindari pembagian dengan nol)
        # Hasil konvolusi grafis, dihitung dengan perkalian matriks adj dan hidden dan dibagi dengan denom
        output = torch.matmul(adj, hidden) / denom # adj: Matriks adjacency yang menggambarkan koneksi antar kata
        if self.bias is not None: # Bias ditambahkan ke output jika diperlukan
            return output + self.bias
        else:
            return output

class REGCN(nn.Module): # adalah kelas utama model ReGCN
    # Inisialisasi kelas dengan matriks embedding dan opsi konfigurasi (`opt`)
    def __init__(self, embedding_matrix, opt):
        super(REGCN, self).__init__()
        self.opt = opt
        self.usebert = self.opt.usebert # Flag untuk menentukan apakah model BERT digunakan
        if self.usebert:
            self.embed = BertModel.from_pretrained('./datasets/bert-base-uncased').requires_grad_(False)
            self.embed.eval() # Inisialisasi model embedding (BERT atau Embedding biasa)
            # Layer LSTM untuk memproses teks
            self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
            self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # Layers konvolusi grafis yang digunakan untuk pemrosesan semantic, syntax, dan dependency
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)

        # Layers konvolusi grafis yang digunakan untuk pemrosesan semantic, syntax, dan dependency
        self.semantic = nn.Sequential(*[GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)  for _ in range(opt.gcn_layer)])
        self.syntax = nn.Sequential(*[GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)  for _ in range(opt.gcn_layer)])
        self.dependency = nn.Sequential(*[GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)  for _ in range(opt.gcn_layer)])
        # Layer fully-connected untuk klasifikasi sentimen
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(opt.dropout) # Layer dropout untuk regularisasi
        # Parameter yang digunakan untuk cross-network
        self.weight = nn.Parameter(torch.FloatTensor(4 * opt.hidden_dim, 4 * opt.hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(4 * opt.hidden_dim))

    # Fungsi untuk menghitung bobot posisi berdasarkan lokasi aspek dalam teks
    def position_weight(self, x, relationX_double_idx, text_len, relationX_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        relationX_double_idx = relationX_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        relationX_len = relationX_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - relationX_len[i]
            for j in range(relationX_double_idx[i,0]):
                weight[i].append(1-(relationX_double_idx[i,0]-j)/context_len)
            for j in range(relationX_double_idx[i,0], relationX_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(relationX_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-relationX_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    # Fungsi untuk melakukan cross-network antara representasi semantic dan syntax
    def cross_network(self,f0,fn):
        fn_weight = torch.matmul(fn,self.weight)
        fl = f0*fn_weight + self.bias + f0
        x = fl[:,:,0:2*self.opt.hidden_dim]
        y = fl[:,:,2*self.opt.hidden_dim:]
        return x,y

    # Fungsi untuk me-mask representasi teks di luar rentang aspek
    def mask(self, x, relationX_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        relationX_double_idx = relationX_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(relationX_double_idx[i,0]):
                mask[i].append(0)
            for j in range(relationX_double_idx[i,0], relationX_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(relationX_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    # Fungsi utama model yang mengambil masukan dan melakukan prediksi sentimen
    def forward(self, inputs):
        # inputs: Masukan model, termasuk indeks teks, indeks aspek, dan matriks adjacency
        text_indices, relationX_indices, left_indices, pmi_adj, cos_adj, dep_adj = inputs
        relationX_len = torch.sum(relationX_indices != 0, dim=-1) # Panjang aspek dan bagian kiri teks
        left_len = torch.sum(left_indices != 0, dim=-1)
        # Indeks awal dan akhir aspek
        relationX_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+relationX_len-1).unsqueeze(1)], dim=1)
        if not self.usebert:
            text_len = torch.sum(text_indices != 0, dim=-1)
            text = self.embed(text_indices) # Hasil embedding teks
            text_out = self.text_embed_dropout(text)
            text_out, (_, _) = self.text_lstm(text_out, text_len) # Hasil dari layer LSTM
            f0 = torch.cat([text_out, text_out], dim=2) # Representasi awal teks yang digabungkan dari text_out
        else:
            text_len = torch.sum(text_indices != 0, dim=-1) - 1
            text, pool = self.embed(text_indices)
            pad0 = torch.zeros((text.shape[0], 1, text.shape[2])).to(self.opt.device)
            text = torch.cat((text[:, 1:, :], pad0), dim=1)
            text_out = self.text_embed_dropout(text)
            text_out, (_, _) = self.text_lstm(text_out, text_len - 2)
            f0 = torch.cat([text_out, text_out], dim=2)
        num_layer = self.opt.num_layer
        f_n = None

        for i in range(num_layer):
            if i == 0:
                # Hasil konvolusi grafis menggunakan matriks adjacency PMI dan cosine
                x_pmi_1 = F.relu(self.gc1(self.position_weight(text_out, relationX_double_idx, text_len, relationX_len), pmi_adj))
                x_pmi_2 = F.relu(self.gc2(self.position_weight(x_pmi_1, relationX_double_idx, text_len, relationX_len), pmi_adj))
                x_cos_1 = F.relu(self.gc3(self.position_weight(text_out, relationX_double_idx, text_len, relationX_len),cos_adj))
                x_cos_2 = F.relu(self.gc4(self.position_weight(x_cos_1, relationX_double_idx, text_len, relationX_len),cos_adj))
                x_s = torch.cat([(x_pmi_2) ,  (x_cos_2)],dim=2)
            else:
                x_d_pmi = F.relu(self.gc1(self.position_weight(x_pmi, relationX_double_idx, text_len, relationX_len), dep_adj))
                x_p_d = F.relu(self.gc2(self.position_weight(x_d_pmi, relationX_double_idx, text_len, relationX_len), dep_adj))
                x_d_cos = F.relu(self.gc3(self.position_weight(x_cos, relationX_double_idx, text_len, relationX_len), dep_adj))
                x_c_d = F.relu(self.gc4(self.position_weight(x_d_cos, relationX_double_idx, text_len, relationX_len), dep_adj))
                f_n = torch.cat([(0.3*x_pmi_2 + x_p_d) ,  (0.3*x_cos_2 + x_c_d)],dim=2)

        # Hasil akhir dari layer konvolusi grafis yang telah di-mask dan di-attention
        x = self.mask(f_n, relationX_double_idx)
        alpha_mat = torch.matmul(x, f0.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, f0).squeeze(1)  #(batch,hidden_dim)
        output = self.dfc(x) # Prediksi sentimen
        return output ,text_indices, alpha