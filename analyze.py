import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as f
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
import  numpy as np
from textblob import TextBlob





data = pd.read_csv('srilanka.csv')


we=data.drop(['Author','date','Location'], axis = 1)  

we['Review'] = we['Review'].astype(str)

#print(we['Review'][1])

we['Review'] = we['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

stop = stopwords.words('english')
we['Review'] = we['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

st = PorterStemmer()
we['Review'] = we['Review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


def senti(x):
    return TextBlob(x).sentiment
we['senti_score'] = we['Review'].apply(senti)
#print(we['senti_score'].str.split(",",expand =True))



df = pd.DataFrame(we)
#df[df==' ']=np.nan
#a =  df['senti_score'].dtype #str.split(',',expand=True)
df['senti'],df['xdr']=zip(*df.senti_score)
#print(df['senti'])
ne=0
pc=0
nn=0
for index, row in df.iterrows():
    if (row["senti"] > 0):
        pc=pc+1
    elif(row["senti"] < 0):
        ne=ne+1
    else:
        nn=nn+1
print ('posivive review: ',pc)
print ('neutral review: ',nn)
print ('negative review: ',ne)
#print(type(df['senti_score']))
#ee=we['date'].value_counts()
#print(we)

#print(f'data1 = {len(we[we["date"] == "2019"])}')

