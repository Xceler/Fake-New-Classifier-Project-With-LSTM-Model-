import re 
import nltk 
import numpy as np 
import pandas as pd 
import zipfile
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.preprocessing.text import one_hot 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential 


#Downloading nltk necessary 
nltk.download('stopwords')
nltk.download('punkt')


#Handling Zipfile
zip_fil = 'fake_or_real_news.csv.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall()


#Handling csv file
data = pd.read_csv('fake_or_real_news.csv')
print(data.head())

#Dropping data for feature and label
x = data.dropna()
x = data.drop('label', axis = 1)
y = data['label']

wordnet = WordNetLemmatizer()
corpus = []  
vocab_size = 5000

#Data Preprocessing 
for i in range(len(x)):

    #For removing punctuation and others things except a-z, A-Z 
    review = re.sub('[^a-zA-Z]', ' ', x['title'][i])

    #Lowercasing
    text = review.lower()

    #Splitting
    txt = text.split()

    #Lemmatization
    txts = [wordnet.lemmatize(word) for word in txt if word not in set(stopwords.words('english'))]

    res  = ' '.join(txts)
    corpus.append(res)


#One Hot Representation 
one_hot_re = [one_hot(words, vocab_size) for words in corpus]

#Embedding Representation 
sent_len = 20
embedded_re = pad_sequences(one_hot_re, padding = 'pre', maxlen = sent_len)

#Modeling LSTM Architecutre 

embedded_feature = 50
model = Sequential([
    Embedding(vocab_size, embedded_feature, input_length = sent_len),
    Dropout(0.02),
    LSTM(100), 
    Dropout(0.02),
    Dense(1, activation = 'sigmoid')
])


#Compiling The model 
model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])

x_final = np.array(embedded_re)
y_final = np.array(y)

#Splitting data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.2, random_state = 42)

#Label Encoding for y_train and y_test

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

#Fit the model 
model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), batch_size= 32)
