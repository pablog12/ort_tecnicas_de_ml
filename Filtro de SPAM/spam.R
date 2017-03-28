library(tm)
library(wordcloud)
library(stringi)
library(SnowballC)

file = "C:/Users/pablo/Documents/Postgrado de Bigdata/Tecnicas de ML/Filtro de SPAM/sms.csv"
dataset = read.csv(file)

##########################
# Creacion del diccionario
##########################

# Texto crudo
txt_raw = dataset$Text

# Remover puntuacion, caracteres indeseables, numeros ...
# Esto es una regex escrita en R, saca puntuacion y basura de la muestra
# Stringi es una funcion que busca el patron definido y lo remplaza.
my_pattern = "[:punct:]|[:cntrl:]|[:digit:]|[-+=$|<>]|[^[:ascii:]]"
my_remove = function(x) stringi::stri_replace_all_regex(x, my_pattern, " ")

print("Removing punctuation, numbers and other symbols ...")
# Esta mapply aplica la funcion que remueve la puntuacion a todas las filas de la muestra.
txt_raw = mapply(my_remove, txt_raw)

# Crear el corpus, es decir, el diccionario para representar los emails.
print("Creating the corpus ...")
txt_corpus = VCorpus(VectorSource(txt_raw))

# Convertir a minusculas
print("Transforming to lower case ...")
txt_corpus = tm_map(txt_corpus, content_transformer(tolower))

# Remover "stopwords", ej., pronombres, preposiciones, ...
print("Removing stop words ...")
txt_corpus = tm_map(txt_corpus, removeWords, stopwords("SMART"))

# Stemming (derivacion): ej., called, calling, caller -> call
print("Stemming ...")
txt_corpus = tm_map(txt_corpus, stemDocument)

# Crear la matriz documento-palabra
print("Creating document-term matrix ...")
txt_dtm = DocumentTermMatrix(txt_corpus)

# Crear el diccionario
print("Creating dictionary ...")
dictionary = unlist(dimnames(txt_dtm)[2])

###################################################
# Construir la representacion por bolsa de palabras
###################################################

print("Creating bag of words representation of documents ...")
sms_txt = as.matrix(txt_dtm)
ndocs = dim(sms_txt)[1]
nwords = dim(sms_txt)[2]
print(toString(c("ndocs = ", ndocs, "nwords = ", nwords)))

##################################
# Inspeccionar el dataset obtenido
##################################

# Calcular la cantidad de ocurrencias de cada palabra
print("Counting words occurrences ...")
word_count = colSums(sms_txt)

# Ejemplo: cuantas ocurrencias tiene la palabra "xxx"
print("Occurrences of ...")
print(word_count[which(dictionary == "xxx")])

# Visualizar el histograma
hist(word_count, main = "histogram of word occurrences", 
     xlab = "number of occurrences", ylab = "frequency", 
     ylim = c(0, nwords))

# Visualizar un grafico de barras
print("Printing bar chart of words with more than 200 occurrences ...")
barplot(word_count[word_count>200], main = "words with more than 200 occurrences", 
        xlab ="word", ylab = "count", 
        names.arg = dictionary[word_count>200], cex.names = 0.8, 
        ylim = c(0,max(word_count)*1.2))

# Visualizar la nube de palabras
print("Showing word cloud ...")
set.seed(17)
wordcloud(dictionary, word_count, min.freq = 100)

###################################################
# Construir el dataset para obtener el clasificador
###################################################

print("Creating the dataframe for learning ...")

# Funcion auxiliar para convertir la clase en 0-1
is_spam = function(x) return(as.integer(x == "spam"))

# Convertir la clase en 0=ham, 1=spam
sms_cls = mapply(is_spam, dataset$Class)

# Hams
print("Ham sms ...")
sms_ham = sms_txt[sms_cls == 0,] 
w_ham = sum(sms_ham)
pr_ham = dim(sms_ham)[1] / ndocs 

print(toString(c("dim = ", dim(sms_ham), "w_ham = ", w_ham, "pr_ham = ", pr_ham)))

# Spams
print("Spam sms ...")
sms_spam = sms_txt[sms_cls == 1,]
w_spam = sum(sms_spam)
pr_spam = dim(sms_spam)[1] / ndocs 

print(toString(c("dim = ", dim(sms_spam), "w_spam = ", w_spam, "pr_spam = ", pr_spam)))

# p(w | ham)
prob_ham = function(w) return(sum(sms_ham[, which(dictionary==w)])/w_ham)

# p(w | spam)
prob_spam = function(w) return(sum(sms_spam[, which(dictionary==w)])/w_spam)

# Matriz de probabilidades condicionales palabra vs ham/spam
print("Creating conditional probability matrix word vs ham/spam ...")
pr = data.frame(ham = mapply(prob_ham, dictionary), spam = mapply(prob_spam, dictionary))

# Logaritmo de x(w) * p(w | y)
my_log = function(x) if (x == 0) return(.Machine$double.xmin) else return(log(x))
logprob = function(x, w) return(sms_txt[x, w]*(my_log(pr$spam[w])-my_log(pr$ham[w])))

# Umbral
b = log(10) + log(pr_spam) - log(pr_ham)

print("Creating sms classifier ...")
L = function(x) return(b + sum(mapply(logprob, x, c(1:nwords))))

