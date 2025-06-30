from os import listdir
from os.path import isfile, join
import numpy as np
input_files = sorted([f for f in listdir('text_clustering') if isfile(join('text_clustering', f))])
class_ids = np.hstack((np.zeros(10), np.ones(10)))
print(class_ids)
print(input_files)

from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


count_vect = CountVectorizer(input='filename')
# рассчитываем признаки, выводим их количество
files = ['./text_clustering/'+ f for f in input_files]
def cluster_by_vectorizer(vect, files):
    print(vect.__name__)
    count_vect = vect(input='filename')
    X_train_counts = count_vect.fit_transform(files)
    print(X_train_counts.shape)
    brc = Birch(n_clusters=2)
    # кластеризуем данные по признакам, выводим ярлыки
    result = brc.fit(X_train_counts)
    predict = brc.predict(X_train_counts)
    print(predict)
    print("1st cluster:")
    cluster1 = [f for i,f in enumerate(input_files) if predict[i] == 0]
    print(cluster1)
    print("2nd cluster:")
    cluster2 = [f for i,f in enumerate(input_files) if predict[i] == 1]
    print(cluster2)

cluster_by_vectorizer(CountVectorizer, files)
"""
[0 0 0 0 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0]
1st cluster:
['confectionery1.txt', 'confectionery10.txt', 'confectionery2.txt', 'confectionery3.txt', 'confectionery5.txt', 'confectionery7.txt', 'confectionery8.txt', 'meet1.txt', 'meet10.txt', 'meet2.txt', 'meet4.txt', 'meet5.txt', 'meet6.txt', 'meet7.txt', 'meet8.txt', 'meet9.txt']
2nd cluster:
['confectionery4.txt', 'confectionery6.txt', 'confectionery9.txt', 'meet3.txt']
"""
print()
cluster_by_vectorizer(TfidfVectorizer, files)
"""[0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 1 1 1 1]
1st cluster:
['confectionery1.txt', 'confectionery10.txt', 'confectionery2.txt', 'confectionery3.txt', 'confectionery4.txt', 'confectionery5.txt', 'confectionery6.txt', 'confectionery7.txt', 'confectionery8.txt', 'confectionery9.txt', 'meet2.txt', 'meet4.txt']
2nd cluster:
['meet1.txt', 'meet10.txt', 'meet3.txt', 'meet5.txt', 'meet6.txt', 'meet7.txt', 'meet8.txt', 'meet9.txt']
"""
print()
cluster_by_vectorizer(HashingVectorizer, files)
"""1st cluster:
['confectionery10.txt', 'confectionery2.txt', 'confectionery3.txt', 'confectionery4.txt', 'confectionery5.txt', 'confectionery6.txt', 'confectionery7.txt', 'confectionery8.txt', 'confectionery9.txt', 'meet1.txt', 'meet10.txt', 'meet2.txt', 'meet3.txt', 'meet4.txt', 'meet5.txt', 'meet6.txt', 'meet7.txt', 'meet8.txt', 'meet9.txt']
2nd cluster:
['confectionery1.txt']"""
print()
"""
Заметим, что при извлечении численных признаков метод TF-IDF дает наилучший результат, изменим файлы(meet2, meet4), чтобы максимизировать угадывание при использовании этого метода векторизации. Тогда при использовании TF-IDF происходит верная кластеризация обучающей выборки

"""