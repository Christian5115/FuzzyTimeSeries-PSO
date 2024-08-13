import streamlit as st
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from natsort import natsorted

import matplotlib.pyplot as plt
import seaborn as sns

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
  PIL.Image.Resampling = PIL.Image

st.write("""# PSO + FTS""")


def get_fitness(X_partikel):
    fitness = {}
    for i in X_partikel.columns:
        hasil = model.forecast(X_partikel[i].values)
        hasil = hasil.dropna()
        mape_ = np.sum(hasil['nilai_abs_err'].values)/len(hasil['nilai_abs_err'])
        fitness[i] =  mape_
        # st.write(f"{i} = {X_partikel[i].values} => {mape_}")
    key_GBEST = min(fitness,key=fitness.get)
    GBEST = X_partikel[key_GBEST].values
    st.write(f"GBEST[{key_GBEST}] => {GBEST} => {fitness[key_GBEST]}")
    return GBEST,fitness[key_GBEST]
def update_v(pbest,gbest,w,c1,c2,r1,r2):
    pbestT = pbest
    gbestT = gbest
    v = w * 0 + c1 * r1 * (pbestT - pbestT) + c2 * r2 * (gbestT - pbestT)
    return v
def update_kecepatan(kecepatan,GBEST,pbest,w,c1,c2,r1,r2):
    for i in range(len(kecepatan)):
        for j in range(len(kecepatan[i])):
            v = update_v(pbest.values[i][j],GBEST[j],w,c1,c2,r1,r2)
            kecepatan[i][j] = v
    return kecepatan

def update_partikel(X_partikel,kecepatan):
    pbest_new = X_partikel_t.values + kecepatan
    pbest_new = pd.DataFrame(pbest_new.T,columns=X_partikel.columns)
    return pbest_new

def pso(iterations,X_partikel,w,c1,c2,r1,r2):
    pbest = X_partikel
    kecepatan_arr = np.zeros(pbest.T.shape)
    var = {}
    for it in range(iterations):
        st.write(f"iterasi => {it+1}")
        GBEST,fitness = get_fitness(pbest)
        kecepatan =  update_kecepatan(kecepatan_arr,GBEST,pbest.T,w,c1,c2,r1,r2)
        pbest = update_partikel(pbest,kecepatan)
        var[fitness] = GBEST
        st.dataframe(pbest)
        st.write()
    return var

class Fitness:

    def __init__(self,data:pd.DataFrame,universe_discource:tuple):
        """
            self.X = numpy array ndim:1
            self.X_test = DataFrame
            uni_disc = tuple
        """
        self.uni_disc  = universe_discource
        self.data = data['Pembukaan'].values
        self.X_test = data
        self.total_interval = 0
        self.interval = []
        self.jumlah_data = []
        self.rank_interval = []
        self.data_interval = []
        self.forecasted = []
        self.MAPE = 0
        self.genes = sorted(list(set(self.data)))
        self.min = min(self.genes)
        self.max = max(self.genes)
        self.total_rank = 0

    #def load_data_from_excel(self, file_path):
    #   self.df = pd.read_excel(file_path)  # Load the entire DataFrame
    #    self.data = self.df['Pembukaan'].tolist()  # Extract 'Pembukaan' values
    #    self.genes = sorted(list(set(self.data)))
    #    self.min = min(self.genes)
    #    self.max = max(self.genes)

    def interval_awal(self):
        self.init_interval()
        data = self.data
        total_interval = self.total_interval
        interval = self.interval
        jumlah_data = self.jumlah_data

        # hitung jumlah data pada tiap interval
        self.hitung_jumlah_data()
        jumlah_data = self.jumlah_data
        list_delete = []

        # buang yang jumlah data interval-nya 0,
        for i in range(len(jumlah_data)):
            if jumlah_data[i] == 0:
                list_delete.append(i)
                jumlah_data[i] = None
                interval[i] = None

        jumlah_data = [i for i in jumlah_data if i is not None]
        interval = [i for i in interval if i is not None]

        # hitung rank tiap interval
        rank_interval = jumlah_data.copy()

        # urutkan descending
        rank_interval.sort(reverse=True)

        # simpan list rank tiap interval
        self.rank_interval = rank_interval
        self.jumlah_data = jumlah_data
        self.interval = interval
        self.total_interval = len(interval)
        self.interval_awal = interval

    def interval_akhir(self):
        jumlah_data = self.jumlah_data
        rank_interval = self.rank_interval

        # banyak rank yang diambil
        total_rank = self.total_rank

        # buang jumlah_data yang ganda, urutkan kembali keys-nya
        rank_temp = list(set(jumlah_data))
        rank_temp.sort(reverse=True)

        # split interval
        # jika rank interval ada di rank temp, maka split sesuai derajat rank temp
        list_split = []
        for i in range(total_rank):
            for key, value in enumerate(rank_interval):
                # KASUS KETIKA VARIASI JUMLAH DATA DIBAWAH TOTAL_RANK
                if i < len(rank_temp) and rank_interval[key] == rank_temp[i]:
                    list_split.append({
                        'nth_rank': i,
                        'nth_interval': key
                    })

        # sort berdasarkan nth interval
        list_split.sort(key=lambda x: x['nth_interval'])

        # do split interval
        self.split_interval(list_split)

        # hitung jumlah data
        self.hitung_jumlah_data()

    def selisih(self, a, b):
        return abs(a - b)

    def defuzzification(self):
        data_interval = self.data_interval
        data = self.data
        forecasted = []

        for count in range(1, len(data_interval)):
            i = data_interval[count-1]['nth_interval']
            j = data_interval[count]['nth_interval']

            if count == 1:
                aj = self.interval[j]
                step = (aj['atas'] - aj['bawah']) / 4
                forecasted.append(aj['bawah'] + 2 * step)
            elif count == 2:
                n_1 = data[count-1]
                n_2 = data[count-2]
                diff_abs = abs(n_1 - n_2) / 2
                forecasted.append(self.rule1(diff_abs, j))
            else:
                n_1 = data[count-1]
                n_2 = data[count-2]
                n_3 = data[count-3]
                diff_diff = (n_1 - n_2) - (n_2 - n_3)
                aj = self.interval[j]
                step = (aj['atas'] - aj['bawah']) / 4
                if j > i and diff_diff >= 0:
                    forecasted.append(self.rule2(diff_diff, count, j))
                elif j > i and diff_diff < 0:
                    forecasted.append(self.rule3(diff_diff, count, j))
                elif j < i and diff_diff >= 0:
                    forecasted.append(self.rule2(diff_diff, count, j))
                elif j < i and diff_diff < 0:
                    forecasted.append(self.rule3(diff_diff, count, j))
                elif j == i and diff_diff >= 0:
                    forecasted.append(self.rule2(diff_diff, count, j))
                elif j == i and diff_diff < 0:
                    forecasted.append(self.rule3(diff_diff, count, j))

        self.forecasted = forecasted

    # 1 -----
    def sub_himpunan(self,X:np.ndarray):
        """
        X:ndarray 1 ndim
        """
        uni_disc =  self.uni_disc
        p1 = np.insert(X,0,uni_disc[0])
        p1 = np.append(p1,uni_disc[1])
        sub_himpunan = pd.DataFrame({
            'Batas Bawah': p1[:-1],
            'Batas Atas': p1[1:],
            'u' : [i for i in range(1,len(X)+2)]
        })
        sub_himpunan["A"] = np.array([f"A{i}" for i in range(1,len(X)+2)])
        return sub_himpunan

    def fuzzyfikasi(self,X:np.ndarray):
        sub_himpunan = self.sub_himpunan(X).values
        def cek_posisi(nilai, data):
            for i, batas in enumerate(data):
                if nilai >= batas[0] and nilai <= batas[1]:
                    return f"A{i+1}"
            return "Nilai tidak berada dalam rentang yang diberikan"
        hasil_a = []
        X_test = self.X_test.copy()
        # data = sub_himpunan.values
        hasil_a = []
        for nilai in X_test['Pembukaan'].values:
            # Memeriksa posisi nilai dalam data
            hasil = cek_posisi(nilai, sub_himpunan)
            hasil_a.append(hasil)
            # Menampilkan hasil
        X_test["A"] = hasil_a
        return X_test
    # -----

    def FLR(self,fuzzyfikasi:pd.DataFrame):
        X_test = fuzzyfikasi
        tanggal = X_test.index
        FLR = X_test['A'].values
        table_FLR = pd.DataFrame({
            'T(current)': tanggal[:-1],
            'T(t+1)': tanggal[1:],
            'F(current)': FLR[:-1],
            'F(t+1)': FLR[1:],
        })
        return table_FLR

    def FLRG(self,FLR:pd.DataFrame):
        # Membuat FLRG
        table_FLR = FLR
        flrg_dict = defaultdict(set)

        for index, row in table_FLR.iterrows():
            current = row['F(current)']
            next_stage = row['F(t+1)']

            if current not in flrg_dict:
                flrg_dict[current] = set()

            if next_stage:
                flrg_dict[current].add(next_stage)

        # Mengurutkan FLRG berdasarkan nilai F(current) menggunakan natsorted
        sorted_flrg = natsorted(flrg_dict.items())

        # Membuat DataFrame baru dari FLRG yang sudah diurutkan
        flrg = pd.DataFrame(sorted_flrg, columns=['F(current)', 'F(t+1)'])
        flrg['F(t+1)'] = flrg['F(t+1)'].apply(lambda x: ','.join(natsorted(x)) if x else '')

        # Menampilkan DataFrame hasilnya
        return flrg

    def nilai_tengah(self,X:np.ndarray):
        sub_himpunan = self.sub_himpunan(X)
        def n(i):
            x = sub_himpunan.loc[sub_himpunan['A'] == i]
            x_bawah = x['Batas Bawah'].values[0]
            x_atas = x['Batas Atas'].values[0]
            return (x_bawah+x_atas)/2
        sub_himpunan["Nilai Tengah"] = sub_himpunan["A"].apply(n)
        return sub_himpunan

    def defuzzifikasi(self,X:np.ndarray,FLR:pd.DataFrame):
        sub_himpunan = self.nilai_tengah(X)
        defuzzifikasi = []
        flrg = self.FLRG(FLR)
        f =  []
        hasil = []
        for i in sub_himpunan["A"].values :
            if i in flrg['F(current)'].values :
                defuz = flrg.loc[flrg['F(current)'] == i]['F(t+1)'].values[0].split(",")
                nilai_tengah = sub_himpunan.loc[sub_himpunan["A"].isin(defuz)]['Nilai Tengah'].values
            else:
                defuz = [i]
                nilai_tengah =  sub_himpunan.loc[sub_himpunan["A"] == i]["Nilai Tengah"].values
            defuzzifikasi.append(defuz)
            f.append(i)
            hasil.append(np.mean(nilai_tengah))
        return pd.DataFrame({"f(t-1)":f,"defuzzifikasi":defuzzifikasi,"hasil":hasil})

    def forecast(self,X:np.ndarray):

        fuzz = self.fuzzyfikasi(X)
        table_FLR = self.FLR(fuzz)

        j = 1
        z_ = {}
        X_test = self.X_test

        defuzzifikasi =  self.defuzzifikasi(X,table_FLR)
        for i in X_test.index:
            if i in table_FLR["T(current)"].values:
                z =  table_FLR.loc[(table_FLR["T(current)"] == i)]["F(current)"].values[0]
            else:
                z =  table_FLR.loc[(table_FLR["T(t+1)"] == i)]["F(current)"].values[0]
                continue
            j+=1
            z = defuzzifikasi.loc[defuzzifikasi['f(t-1)']==z]["hasil"].values[0]
            z_[j] = z

        hasil_peramalan = []
        for i in range(1,len(X_test)+1):
            aktual = X_test.iloc[i-1]["Pembukaan"]
            X = X_test.index[i-1].date().strftime("%b-%Y")

            if i == 1:
                p_ = [X,aktual,None,None,None]
            else:
                pred =  z_[i]
                err = (aktual-pred)**2
                absolute_error =  abs(aktual-pred)/aktual
                p_ = [X,aktual,pred,err,absolute_error]
            hasil_peramalan.append(p_)
        hasil_peramalan = pd.DataFrame(hasil_peramalan,columns=[["tanggal","Pembukaan","prediksi","error","nilai_abs_err"]])
        return hasil_peramalan

    def getMAPE(self,X:np.ndarray):
        data = X
        forecasted = self.forecasted
        error = 0

        for i in range(min(len(forecasted) - 1, len(data) - 1)):
            error += abs((data[i + 1] - forecasted[i]) / data[i + 1])

        self.MAPE = error / len(forecasted)

    def rule1(self, diff_abs, j):
        aj = self.interval[j]
        step = (aj['atas'] - aj['bawah']) / 4
        half = 2 * step

        if diff_abs > half:
            return aj['bawah'] + 3 * step
        elif diff_abs == half:
            return aj['bawah'] + 2 * step
        else:
            return aj['bawah'] + step

    def rule2(self, diff_diff, n, j):
        diff_diff = abs(diff_diff)
        aj = self.interval[j]
        n_1 = self.data[n - 1]
        power = diff_diff * 2
        div = diff_diff / 2
        step = (aj['atas'] - aj['bawah']) / 4

        if aj['bawah'] <= power + n_1 <= aj['atas'] or aj['bawah'] <= n_1 - power <= aj['atas']:
            return aj['bawah'] + 3 * step
        elif aj['bawah'] <= div + n_1 <= aj['atas'] or aj['bawah'] <= n_1 - div <= aj['atas']:
            return aj['bawah'] + step
        else:
            return aj['bawah'] + 2 * step

    def rule3(self, diff_diff, n, j):
        diff_diff = abs(diff_diff)
        aj = self.interval[j]
        n_1 = self.data[n - 1]
        power = diff_diff * 2
        div = diff_diff / 2
        step = (aj['atas'] - aj['bawah']) / 4

        if aj['bawah'] <= div + n_1 <= aj['atas'] or aj['bawah'] <= n_1 - div <= aj['atas']:
            return aj['bawah'] + step
        elif aj['bawah'] <= power + n_1 <= aj['atas'] or aj['bawah'] <= n_1 - power <= aj['atas']:
            return aj['bawah'] + 3 * step
        else:
            return aj['bawah'] + 2 * step

    def init_interval(self):
        interval = []
        genes = self.genes.copy()
        genes.insert(0, self.min)
        genes.append(self.max)

        for i in range(len(genes) - 1):
            interval.append({
                'bawah': genes[i],
                'atas': genes[i + 1]
            })

        self.interval = interval

    def hitung_jumlah_data(self):
        data = self.data
        total_interval = self.total_interval
        jumlah_data = [0] * total_interval
        interval = self.interval

        data_interval = []

        for i in range(len(data)):
            for j in range(total_interval):
                if (interval[j+1]['bawah']) - 1 <= data[i] <= interval[j]['atas'] + 1:
                    jumlah_data[j] += 1
                    data_interval.append({
                        'data': data[i],
                        'nth_interval': j
                    })
                    break
    def split_interval(self, list_split):
        interval = self.interval
        count = 0

        for item in list_split:
            nth_interval = item['nth_interval']

            if item['nth_rank'] == 0:
                self.split(4, nth_interval + count)
                count += 3
            elif item['nth_rank'] == 1:
                self.split(3, nth_interval + count)
                count += 2
            elif item['nth_rank'] == 2:
                self.split(2, nth_interval + count)
                count += 1

        self.total_interval = len(self.interval)

    def split(self, n_split, idx_interval):
        interval = self.interval
        idx = idx_interval
        bawah = interval[idx]['bawah']
        atas = interval[idx]['atas']
        step = atas - bawah
        split_step = step / n_split

        data_insert = []

        for i in range(n_split):
            data_insert.append({
                'bawah': bawah + (split_step * i),
                'atas': bawah + (split_step * (i + 1))
            })

        interval[idx_interval] = data_insert[0]
        count = idx_interval

        for i in range(1, n_split):
            interval.insert(count + i, data_insert[i])

        self.interval = interval

upload_file = st.file_uploader("Pick a File")

with st.sidebar:
    iterasi     = st.number_input("Nilai Iterasi:", min_value=1,max_value=1000,value=20, placeholder="Input angka disini..",step=1)
    partikel    = st.number_input("Banyaknya Partikel:",min_value=1,max_value=100,value=10,placeholder="Input angka disini..",step=1)
    dimensi     = st.number_input("Dimensi Partikel:",min_value=1,max_value=100,value=5,placeholder="Input angka disini..",step=1)
    c1          = st.slider('Nilai C1: ',0.0,3.0,(0.3))
    c2          = st.slider('Nilai C2: ',0.0,3.0,(0.3))
    w           = st.slider('Nilai Bobot Inersia: ',0.0,1.0,(0.3))

#st.write('Nilai Iterasi: ', iterasi)
#st.write('Banyaknya Partikel: ', partikel)
#st.write('Dimensi Partikel: ', dimensi)
#st.write('Nilai C1: ', c1)
#st.write('Nilai C2: ', c2)
#st.write('Nilai Bobot Inersia: ', w)

if upload_file is not None:
    df = pd.read_excel(upload_file)
    df['Tanggal'] =  pd.to_datetime(df['Tanggal'])
    df = df.set_index('Tanggal')
    df.head()

if st.button("Start"):
    if upload_file is None:
        st.write("Pls attach file")
    if iterasi <= 0 or partikel <= 0 or dimensi <= 0 or c1 <= 0 or c2 <= 0 or w <= 0:
        st.write("Please input valid number")
    else:
        X = df['Pembukaan'].values

        X_test = df.loc[(df.index >= '2021-10-04') & (df.index <= '2023-10-02')]

        min1 = np.min(X)
        max1 = np.max(X)

        st.write(min1)
        st.write(max1)

        list1 = {}

        for i in range (0,partikel):
            list1['P'+str(i+1)] = []

            list1['P'+str(i+1)] = random.sample(range(int(min1),int(max1)),dimensi)

            list1['P'+str(i+1)].sort()


        # Partikel Awal
        X_partikel = pd.DataFrame(list1)
        kecepatan_ = np.zeros(X_partikel.T.shape)

        X_max_partikel =  np.max(X_partikel.values)
        X_min_partikel =  np.min(X_partikel.values)
        X_partikel_t =  X_partikel.T

        D1 =  105
        D2 = 159
        r1 = 0.3
        r2 = 0.3
        universe_discourse = X_min_partikel-D1,X_max_partikel+D2

        Xmin,Xmax = universe_discourse
        k = 0.6
        vmax = k*(Xmax-Xmin)/2
        vmin = -vmax

        st.write("Nilai Himpunan Semesta: ",universe_discourse)
        st.write("Partikel Awal: ")
        st.write(X_partikel.T)

        model = Fitness(X_test,universe_discourse)
        st.write(pd.DataFrame(model.data))

        result = pso(iterasi,X_partikel,w,c1,c2,r1,r2)

        ex =  list(result.values())[0]
        GBEST_I = pd.DataFrame(list(result.values()),columns=[f"X{i}" for i in range(1,len(ex)+1)])
        st.write(GBEST_I)

        fitness_ = pd.DataFrame({'id_GBEST':GBEST_I.index,'fitness':[i for i in result.keys()]})

        ID_GBEST_RESULT = fitness_[fitness_['fitness'] == np.min(fitness_['fitness'])]
        st.write(ID_GBEST_RESULT)

        GBEST_RESULT = GBEST_I.iloc[ID_GBEST_RESULT['id_GBEST'].values[0]].values
        st.write(GBEST_RESULT)

        hasil_peramalan =  model.forecast(GBEST_RESULT)
        st.dataframe(hasil_peramalan)

        h =  hasil_peramalan.dropna()
        mape =  np.sum(h['nilai_abs_err'].values)/len(h['nilai_abs_err'])
        st.write(mape)

        fig = plt.figure(figsize=(20, 6))
        plt.plot(X_test.index, hasil_peramalan['Pembukaan'].values, label='Observasi', marker='o')
        plt.plot(X_test.index, hasil_peramalan['prediksi'].values, label='Prediksi', linestyle='--', marker='o')  # Dimulai dari indeks ke-2 untuk mengatasi nilai prediksi yang kurang
        plt.title(f'Fuzzy Time Series Prediction MAPE : {mape.round(3)}')
        plt.xlabel('Tanggal')
        plt.ylabel('Pembukaan')
        plt.legend()
        fig = fig.figure
        st.pyplot(fig)


