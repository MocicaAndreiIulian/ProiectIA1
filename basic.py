
import random
import nltk
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import time

start = time.time()

def tokenize(text):

    text_lower = text.lower()
    tkn = re.sub(r"http\S+", "LINK", text_lower)
    #tkn0 = re.sub(r"l'\S+","ART",tkn)
    tkn1 = re.sub(r"@\S+", "UTILIZATOR", tkn)
    #tkn2 = ' '.join(word.strip(string.punctuation) for word in tkn1.split())
    #tkn2 = re.sub(r"[,.;?&$():_*!=+'\"]", " ", tkn1)
    tkn2 = re.sub(r"#\S+", " ", tkn1)
    tkn3 = re.sub(r"\d+", " ", tkn2)
    #tkn4 = re.sub(r"[,.;?&$():_*!=+#'\"]", " ", tkn3)
    #tkn4 = re.sub(r"[:,;&!?.]", " ", tkn3)
    #tkn4 = re.sub(r"[^\w]"," ",tkn3)
    #tkn4 = re.sub(r"[,.;?&]", " ", tkn3)
    #tkn4 = re.sub(r"\W", " ", tkn3)
    #tkn4 = " ".join(punct.strip(string.punctuation) for punct in tkn3.split())
    #tkn12 = tokenize(tkn4)
    tkn5 = " ".join([cuvant for cuvant in tkn3.split() if len(cuvant) > 3])
    tkn6 = " ".join([cuvant2 for cuvant2 in tkn5.split() if len(cuvant2) < 17])
    #tkn7 = " ".join(punct.strip(string.punctuation) for punct in tkn6.split())
    #tkn2 = "".join([semn_punctuatie.lower() for semn_punctuatie in tkn if semn_punctuatie not in string.punctuation])
    #tkn6 = re.sub(r"[^\w]"," ",tkn5)
    token = TweetTokenizer(reduce_len=True).tokenize(tkn6)
    #token = WhitespaceTokenizer().tokenize(tkn6)
    #return nltk.WordPunctTokenizer().tokenize(tkn6)
    return token

#Construirea dictionarului
def get_representation(toate_cuvintele, how_many):

    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


#Toate cuvintele din dictionar
def get_corpus_vocabulary(corpus):

    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter

#Numara de cate ori apare un cuvant
def text_to_bow(text, wd2idx):

    features = np.zeros(len(wd2idx))
    for cuvant in tokenize(text):
        if cuvant in wd2idx:
            idx = wd2idx[cuvant]
            features[idx] += 1
    return features

#Numara de cate ori apar fiecare cuvant (folosind text_to_bow)
def corpus_to_bow(corpus, wd2idx):

    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        bow = text_to_bow(text, wd2idx)
        all_features[i, :] = bow

    return all_features
def cross_validate(k, data, labels):

    chunk_size = int(len(labels) / k)
    indici = np.arange(0,len(labels))
    random.shuffle(indici)

    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        right_side = indici[i+chunk_size:]
        left_side = indici[0:i]
        train_indici = np.concatenate([left_side, right_side])
        train = data[train_indici]
        valid = data[valid_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid


def matrice_confuzie(stiute, pred):

    rezultat = np.zeros((2, 2)) #2 clase 1 si 0

    for i in range(len(stiute)):
        rezultat[stiute[i]][pred[i]] += 1

    return rezultat



###COD

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
toate_cuvintele = get_corpus_vocabulary(train_df['text'])
wd2idx, idx2wd = get_representation(toate_cuvintele, len(toate_cuvintele))
data = corpus_to_bow(train_df['text'], wd2idx)
labels = train_df['label']
test_data = corpus_to_bow(test_df['text'], wd2idx)

'''print(len(corpus_to_bow(test_df['text'], wd2idx)))
print(len(get_corpus_vocabulary(test_df['text'])))
print(get_corpus_vocabulary(test_df['text']))'''
#Comparare cu 36
'''clf_MNB = MultinomialNB()
clf_MNB.fit(data, labels)
test_22 = pd.read_csv('Submisie_Kaggle_36.csv')
labels_22 = test_22['label']
print("Acuratete:",metrics.accuracy_score(labels_22, clf_MNB.predict(test_data)))
print("F1:",metrics.f1_score(labels_22, clf_MNB.predict(test_data)))
print("Recall:",metrics.recall_score(labels_22, clf_MNB.predict(test_data)))
print(labels.mean()*5000)'''
#CROSS
clf_MNB = MultinomialNB()

scoruri = []
matrice = np.zeros((2,2))
for train, valid, y_train, y_valid in cross_validate(10,data,labels):
    clf_MNB.fit(train,y_train)
    pred = clf_MNB.predict(valid)
    scor = fbeta_score(y_valid, pred, beta =1)
    matrice += matrice_confuzie(list(y_valid), list(pred))
    print(scor)
    scoruri.append(scor)

print("Media scorurilor in urma cross validation")
print(np.mean(scoruri))
print("Diferenta medie este:")
print(np.std(scoruri))
print("Matricea este")
print(matrice)

#CONFUZIE
'''date_invatare = data[:4500, :]
date_testare = data[4500:5000, :]
labels_invatare = train_df['label'][:4500]
labels_testare = train_df['label'][4500:5000]
clf_MNB2 = MultinomialNB()
clf_MNB2.fit(date_invatare, labels_invatare)
print("Matricea de confuzie")
print(matrice_confuzie(list(labels_testare), list(clf_MNB2.predict(date_testare))))
tp, fp, fn, tn = matrice_confuzie(list(labels_testare), list(clf_MNB2.predict(date_testare))).ravel()
Acuratete = (tn+tp)*100/(tp+tn+fp+fn)
print("Acuratete conform matricei",Acuratete)'''
#print(matrice_confuzie(list(labels_testare), list(clf_MNB.predict(date_testare))).ravel())
#result = confusion_matrix(list(labels_testare), list(clf_MNB.predict(date_testare)))
#print(result)
stop = time.time()
print("Durata", stop - start)

# 3 CLASIFICATORI (NOT GOOD)
'''clf_CNB = ComplementNB()
clf_CNB.fit(data, labels)

clf_MNB = MultinomialNB()
clf_MNB.fit(data, labels)

clf_KNE = KNeighborsClassifier(5)
clf_KNE.fit(data, labels)

clf_pred = np.zeros(1000)

for i in range(1000):
    if clf_CNB.predict(test_data)[i] == clf_MNB.predict(test_data)[i]:
        clf_pred[i] = clf_CNB.predict(test_data)[i]
    elif clf_CNB.predict(test_data)[i] == clf_KNE.predict(test_data)[i]:
        clf_pred[i] = clf_CNB.predict(test_data)[i]
    elif clf_KNE.predict(test_data)[i] == clf_MNB.predict(test_data)[i]:
        clf_pred[i] = clf_MNB.predict(test_data)[i]
    else:
        clf_pred[i] = clf_MNB.predict(test_data)[i]

#print(clf_pred)


testID = np.arange(5001, 6001)
#predictedLabels = np.ones(1000)


np.savetxt("D:\Facultate_An3\InteligentaArtificiala\Proiect\Submisie_Kaggle_17.csv", np.stack((testID, clf_pred)).T,
      fmt = "%d", delimiter=',', header="id,label", comments='')'''


#NAIVE BAYES
'''clf = MultinomialNB()
clf.fit(data, labels)
clf.predict(test_data)'''
#print(clf.predict(test_data).sum())


#date_invatare = data[:4000, :]
#date_testare = data[4000:5000, :]
#labels_invatare = train_df['label'][:4000]
#labels_testare = train_df['label'][4000:5000]

'''
clf_test_CNB = ComplementNB()
clf_test_CNB.fit(date_invatare, labels_invatare)

clf_test_MNB = MultinomialNB()
clf_test_MNB.fit(date_invatare, labels_invatare)

clf_test_KNE = KNeighborsClassifier(5)
clf_test_KNE.fit(date_invatare, labels_invatare)

clf_pred = np.zeros(1000)

for i in range(1000):
    if clf_test_CNB.predict(date_testare)[i] == clf_test_MNB.predict(date_testare)[i]:
        clf_pred[i] = clf_test_CNB.predict(date_testare)[i]
    elif clf_test_CNB.predict(date_testare)[i] == clf_test_KNE.predict(date_testare)[i]:
        clf_pred[i] = clf_test_CNB.predict(date_testare)[i]
    elif clf_test_KNE.predict(date_testare)[i] == clf_test_MNB.predict(date_testare)[i]:
        clf_pred[i] = clf_test_MNB.predict(date_testare)[i]
    else:
        clf_pred[i] = clf_test_MNB.predict(date_testare)[i]

print("Acuratete:",metrics.accuracy_score(labels_testare, clf_pred))
print("F1:",metrics.f1_score(labels_testare, clf_pred))
print("Recall:",metrics.recall_score(labels_testare, clf_pred))'''


'''clf_test = MultinomialNB()
clf_test.fit(date_invatare, labels_invatare)


print("Acuratete:",metrics.accuracy_score(labels_testare, clf_test.predict(date_testare)))
print("F1:",metrics.f1_score(labels_testare, clf_test.predict(date_testare)))
print("Recall:",metrics.recall_score(labels_testare, clf_test.predict(date_testare)))'''


#print(len(get_corpus_vocabulary(train_df['text'])))


'''predictii = np.zeros(1000)
for i in range(1000):
    if clf_test.predict(date_testare)[i] == labels_testare[4000+i]:
        predictii[i] = 1

print(predictii.mean())'''




#print(get_representation(toate_cuvintele, 10))
#print(get_corpus_vocabulary(train_df['text']))
#print(get_corpus_vocabulary(test_df['text']))
'''for vad in range(1, 1000):
    print(test_df['text'][vad])
    print(tokenize(test_df['text'][vad]))'''

'''
j=0
for i in clf1.predict(test_data):
    if clf1.predict(test_data)[i] == 0:
      j +=1

print(j/1000)
'''

#print(clf.predict(test_data))
#print(clf.predict(test_data).mean())


#print(clf.predict(test_data)[3])

#PRINTARE IN FISIER
#testID = np.arange(5001, 6001)
#predictedLabels = np.ones(1000)


#np.savetxt("D:\Facultate_An3\InteligentaArtificiala\Proiect\Submisie_Kaggle_38.csv", np.stack((testID, clf.predict(test_data))).T,
   #fmt = "%d", delimiter=',', header="id,label", comments='')







