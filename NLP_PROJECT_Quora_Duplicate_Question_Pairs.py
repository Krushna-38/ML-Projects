#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('questions.csv')


# In[3]:


new_df = df.sample(30000,random_state=2)


# In[4]:


new_df.head()


# In[5]:


def preprocess(q):
    
    q = str(q).lower().strip()
    
    #Replace certain special characters with their string equivalents
    q = q.replace('%', 'percent')
    q = q.replace('$', 'dollar')
    q = q.replace('₹', 'rupee')
    q = q.replace('€', 'euro')
    q = q.replace('@', 'at')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    #Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)

    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    #Decontracting words
    contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
    
    q_decontracted = []
    
    for word in q.split():
        if word in contractions:
            word = contractions[word]
            
        q_decontracted.append(word)
        
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'11", " will")
    
    #Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    #Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ',q).strip()
    
    return q


# In[6]:


preprocess("I've already| wasn't <b>done</b>?")


# In[7]:


new_df['question1'] = new_df['question1'].apply(preprocess)
new_df['question2'] = new_df['question2'].apply(preprocess)


# In[8]:


new_df.head()


# In[9]:


new_df['q1_len'] = new_df['question1'].str.len()
new_df['q2_len'] = new_df['question2'].str.len()


# In[10]:


new_df['q1_num_words'] = new_df['question1'].apply(lambda row: len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row: len(row.split(" ")))
new_df.head()


# In[11]:


def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)


# In[12]:


new_df['word_common'] = new_df.apply(common_words, axis=1)
new_df.head()


# In[13]:


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))


# In[14]:


new_df['word_total'] = new_df.apply(total_words, axis=1)
new_df.head()


# In[15]:


new_df['word_share'] = round(new_df['word_common']/new_df['word_total'], 2)
new_df.head(11)


# In[16]:


#Advanced Features
from nltk.corpus import stopwords

def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001
    
    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    #Converting the sentence into TOkens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    #Get the non-stopwordsin questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopword in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    #Get the common non_stopwords from question pairs
    common_word_count = len(q1_words.intersection(q2_words))
    
    #Get the common stopwords from question pairs
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    #Get the common  Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_word_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_word_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    #Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    #First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features
    


# In[17]:


token_features = new_df.apply(fetch_token_features, axis=1)

new_df["cwc_min"]       = list(map(lambda x: x[0], token_features))
new_df["cwc_max"]       = list(map(lambda x: x[1], token_features))
new_df["csc_min"]       = list(map(lambda x: x[2], token_features))
new_df["csc_max"]       = list(map(lambda x: x[3], token_features))
new_df["ctc_min"]       = list(map(lambda x: x[4], token_features))
new_df["ctc_max"]       = list(map(lambda x: x[5], token_features))
new_df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
new_df["first_word_eq"] = list(map(lambda x: x[7], token_features))



# In[18]:


new_df.head()


# In[19]:


import distance 

def fetch_length_features(row):
    q1 = row["question1"]
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    #Converting the sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    #Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of Both Questions
    length_features[1] = (len(q1_tokens) +  len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) +1)
    
    return length_features

    


# In[20]:


length_features = new_df.apply(fetch_length_features, axis=1)

new_df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
new_df['mean_len'] = list(map(lambda x: x[1], length_features))
new_df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))


# In[21]:


new_df.head()


# In[22]:


# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    #Fuzzy_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    #Fuzzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    
    #Token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    #token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    
    return fuzzy_features


# In[23]:


fuzzy_features = new_df.apply(fetch_fuzzy_features, axis=1)

#Creating new feature columns for fuzzy features
new_df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
new_df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
new_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
new_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


# In[24]:


print(new_df.shape)
new_df.head()


# In[25]:


sns.pairplot(new_df, vars=['ctc_min', 'cwc_min', 'csc_min'], hue='is_duplicate')


# In[26]:


sns.pairplot(new_df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']], hue='is_duplicate')


# In[27]:


sns.pairplot(new_df[['last_word_eq', 'first_word_eq', 'is_duplicate']], hue='is_duplicate')


# In[28]:


sns.pairplot(new_df[['mean_len', 'abs_len_diff', 'longest_substr_ratio', 'is_duplicate']], hue='is_duplicate')


# In[29]:


sns.pairplot(new_df[['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio', 'is_duplicate']], hue='is_duplicate')


# In[30]:


#Using TSNE for Dimensionality reduction for 15 features(Generated after cleaning the data) to 3 dimention
from sklearn.preprocessing import MinMaxScaler

# Select only the numeric columns
numeric_columns = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 'mean_len', 'abs_len_diff', 'longest_substr_ratio', 'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']
numeric_df = new_df[numeric_columns]

# Scale the numeric data
x = MinMaxScaler().fit_transform(numeric_df)

# Now x contains the scaled numeric data
y = new_df['is_duplicate'].values


# In[31]:


from sklearn.manifold import TSNE

tsne2d = TSNE(
    n_components=2,
    init='random', #PCA
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(x)


# In[32]:


import matplotlib.pyplot as plt

# Extracting x and y coordinates from the t-SNE output
x_tsne = tsne2d[:, 0]
y_tsne = tsne2d[:, 1]

# Plotting the t-SNE plot
plt.figure(figsize=(10, 8))
plt.scatter(x_tsne, y_tsne, c=y, cmap='viridis')
plt.title('t-SNE Plot of Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Target Variable')
plt.grid(True)
plt.show()


# In[33]:


tsne3d = TSNE(
    n_components=3,
    init='random', #PCA
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(x)


# In[44]:


import plotly.graph_objs as go
from plotly.offline import iplot

trace1 = go.Scatter3d(
    x=tsne3d[:,0],
    y=tsne3d[:,1],
    z=tsne3d[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color= y,
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3d embedding with engineered features')
fig=dict(data=data, layout=layout)
iplot(fig, filename='3DBubble.html')



# In[36]:


ques_df= new_df[['question1','question2']]
ques_df.head()


# In[37]:


final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
print(final_df.shape)
final_df.head()


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
#merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[39]:


temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[40]:


final_df = pd.concat([final_df, temp_df], axis=1)
print(final_df.shape)
final_df.head()


# In[41]:


from sklearn.model_selection import train_test_split


# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(final_df.iloc[:, 1:].values,  # Features
                                                    final_df.iloc[:, 0].values,    # Target variable
                                                    test_size=0.2, 
                                                    random_state=2)


# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy_score(y_test, y_pred)


# In[43]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred1 = xgb.predict(x_test)
accuracy_score(y_test,y_pred1)


# In[45]:


from sklearn.metrics import confusion_matrix


# In[46]:


#FOr random forest model
confusion_matrix(y_test, y_pred)


# In[47]:


#for xgboost model
confusion_matrix(y_test,y_pred1)


# In[48]:


q1 = 'What is the capital of India'
q2 = 'What is the current capital of India'


# In[49]:


def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


# In[50]:


def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


# In[51]:


def test_fetch_token_features(q1,q2):
    
    SAFE_DIV = 0.0001
    
    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    #Converting the sentence into TOkens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    #Get the non-stopwordsin questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopword in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    #Get the common non_stopwords from question pairs
    common_word_count = len(q1_words.intersection(q2_words))
    
    #Get the common stopwords from question pairs
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    #Get the common  Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_word_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_word_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    #Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    #First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features
    
    


# In[52]:


def test_fetch_length_features(q1,q2):

    length_features = [0.0]*3
    
    #Converting the sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    #Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of Both Questions
    length_features[1] = (len(q1_tokens) +  len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) +1)
    
    return length_features


# In[53]:


def test_fetch_fuzzy_features(q1,q2):

    fuzzy_features = [0.0]*4
    
    #Fuzzy_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    #Fuzzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    
    #Token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    #token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    
    return fuzzy_features
    


# In[54]:


def query_point_creator(q1,q2):
    input_query = []
    
    #Preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    #Fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(test_common_words(q1,q2))
    input_query.append(test_total_words(q1,q2))
    input_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))
    
    #Fetch token features
    token_features = test_fetch_token_features(q1,q2)
    input_query.extend(token_features)
    
    #Fetch length based features
    length_features = test_fetch_length_features(q1,q2)
    input_query.extend(length_features)
    
    #Fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1,q2)
    input_query.extend(fuzzy_features)
    
    #bow feature for q1
    q1_bow = cv.transform([q1]).toarray()
    
    #bow feature for q2
    q2_bow = cv.transform([q2]).toarray()
    
    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))


# In[59]:


q1 = 'What is the capital of India'
q2 = 'What is the current capital of China'


# In[60]:


rf.predict(query_point_creator(q1,q2))


# In[ ]:




