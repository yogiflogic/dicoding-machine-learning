import sklearn
from sklearn import datasets

# Library sklearn menyediakan dataset iris yakni sebuah dataset yang umum digunakan untuk masalah klasifikasi. Dataset ini memiliki jumlah 150 sampe
# load iris dataset
iris = datasets.load_iris()

# pisahkan atribut dan label pada iris dataset
x=iris.data
y=iris.target

# Train_test_split mengembalikan 4 nilai yaitu, atribut dari train set, atribut dari test set, target dari train set, dan target dari test set
from sklearn.model_selection import train_test_split
 
# membagi dataset menjadi training dan testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# menghitung panjang/jumlah data pada x_test
# Pada tahap ini dataset kita telah siap dipakai untuk pelatihan model machine learning
len(x_test)