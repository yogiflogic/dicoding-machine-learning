# https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.MinMaxScaler.html

from sklearn.preprocessing import MinMaxScaler
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]

scaler = MinMaxScaler()

# Fungsi fit() dari objek MinMaxSclaer adalah fungsi untuk menghitung nilai minimum dan maksimum pada tiap kolom.
scaler.fit(data)

print(scaler.transform(data))
