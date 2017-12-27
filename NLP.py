
# coding: utf-8

# # Estudando NLP
# 
# O que eu pretendo usar: CSTNews Corpus
# 
# 
# NLTK;
# 
# Portuguese Examples: http://www.nltk.org/howto/portuguese_en.html

# In[1]:


import nltk


# # 1. Tokenização
# 
# ## 1.1 Tokenização em setenças
# 
# Token é um pedaço de um todo, então: 
# 
# - uma palavra é um token em uma sentença;
# - uma sentença é um token em um paragrafo.
# 
# Logo separar um paragrafo em sentenças é tokenizar sentenças.

# In[2]:


from nltk.tokenize import sent_tokenize

f = open('sample.txt')
dataset = f.read()

sentence_tokenized = sent_tokenize(dataset)
print(sentence_tokenized)


# In[3]:


len(sentence_tokenized)


# ### Tokenização em Português!

# In[4]:


import nltk.data
portuguese_tokenizer = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')
portuguese_tokenizer.tokenize(dataset)


# ## 1.2 Tokenização em palavras

# In[5]:


from nltk.tokenize import word_tokenize

first_sentence_word_tokenized = word_tokenize(sentence_tokenized[0])
print(first_sentence_word_tokenized)## 1.3 Tokenização com expressão regular


# ## 1.3 Tokenização com expressão regular
# 
# Por exemplo para pegar apenas palavras em um texto

# In[6]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+")

first_sentence_word_tokenized_without_punctuation = tokenizer.tokenize(sentence_tokenized[0])
print(first_sentence_word_tokenized_without_punctuation)


# ## 1.4 Treinando um tokenizador de sentenças
# 
# O tokenizador do NLTK é para uso geral, nem sempre é a melhor opção para textos, dependendo da formatação do texto. 

# In[7]:


from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext

text = webtext.raw('overheard.txt')
sent_tokenizer = PunktSentenceTokenizer(text)


# In[8]:


sents = sent_tokenizer.tokenize(text)
sents[0:10]


# Como podemos observador, as sentenças são separadas (tokenizadas) em conversas, por que o `PunktSentenceTokenizer` usa um algorítmo de unsupervised learning para aprender o que constitui uma quebra de sentença. É não-supervisionado por que você não precisa dar nenhum texto para treinamento do algoritmo, apenas o texto em si.

# ## 1.5 Filtrando stopwords
# 
# Stopwords são palavras que geralmente não contribuem para o significado de uma sentença.

# In[9]:


from nltk.corpus import stopwords

portuguese_stops = set(stopwords.words('portuguese'))
words = first_sentence_word_tokenized_without_punctuation

words_without_stop = [word for word in words if word not in portuguese_stops]
print(words_without_stop)


# # 2. Substituindo e corrigindo palavras
# 
# ## 1.1 Stemming 
# 
# Stemming é a técnica que remove os afixos das palavras, deixando apenas seu radical, existe uma versão em Português que é `RSLPStemmer`

# In[10]:


from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
stem_acidente = stemmer.stem(words_without_stop[1]) #acidente
print(stem_acidente)


# Outro stemmer que pode ser usado é o SnowballStemmer, que tem opção pt-br:

# In[11]:


from nltk.stem import SnowballStemmer

stem_acidente = SnowballStemmer('portuguese')
stem_acidente.stem(words_without_stop[1]) #acidente


# Os resultados foram diferentes como podemos notar, então vale ver qual a aplicação, para decidir qual é o melhor.

# ## 3. Transformando texto raw no formato do NLTK
# 
# NLTK tem seu formato de texto padrão, para converter o que é preciso ser feito:
# 
# - Tokenizar o texto em palavras
# - Usar método de conversão para `nltk.txt``

# ### 3.1 Tokenizando em palavras

# In[12]:


text_tokenized = word_tokenize(dataset)
type(text_tokenized)


# ### 3.2 Convertendo

# In[13]:


text_format = nltk.Text(text_tokenized)
type(text_format)


# By transforming a raw text into a text we can use more NLTK features

# ### Análise de estrutura de texto
# 
# A concordância nos permite ver palavras em contexto.

# In[14]:


text_format.concordance('acidente')


# In[15]:


text_format.similar('um')


# In[16]:


text_format.collocations()


# ## 4. Convertendo palavras combinando com expressões regulares
# 
# Essa combinação é boa para converter palavras que foram diminuídas e problemas de linguagem. 
# 
# Por exemplo: 'Pq' > 'porque', 'Coé' > 'Qual é'

# In[17]:


import re
import unicodedata 

replacement_patterns = [
    (r'pq', 'porque'),
    (r'coe', 'qual e'),
    (r'vc', 'voce'),
]
    
def normalize_string(string):
    if isinstance(string, str):
        nfkd_form = unicodedata.normalize('NFKD', string.lower())
        return nfkd_form.encode('ASCII', 'ignore').decode('utf-8')

def replace(text, patterns=replacement_patterns):
    s = normalize_string(text)
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    for (pattern, repl) in patterns:
        s = re.sub(pattern, repl, s)
    return s


# In[21]:


print(replace("Coé, pq vc fez isso?"))


# ### Extra: Enchant
# 
# Existe uma biblioteca que faz correção ortográfica chamada (Enchant)[http://pythonhosted.org/pyenchant/tutorial.html], mas ela não possui dicionário em português:

# In[22]:


import enchant
print(enchant.list_languages())


# Análisando a frequência relativa de palavras

# In[19]:


from nltk import FreqDist

fd1 = FreqDist(text_format)
print(dict(fd1))

