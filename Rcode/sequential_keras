library(dplyr)

sampelreview_top10 <- read.csv("~/1005Inference/sampelreview_top10.csv")
sampelreview_top10$top<- ifelse( sampelreview_top10$n_review>100, 1, 0)
dt<-sampelreview_top10 %>% 
  group_by(reviewerID,top) %>% 
  summarise(text1 = paste0(reviewText, collapse = "")) 
#now dt is the training and test sample 

t<-keras::text_tokenizer(num_words = 100,
      filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n")

train_ind <- sample(seq_len(nrow(dt)), size = 1500)

train <- dt[train_ind, ]
test <- dt[-train_ind, ]



reviewtext<-as.character(dt$text1)
keras::fit_text_tokenizer(t, reviewtext1)
t$document_count
t$word_index
texts_sequence <- texts_to_sequences(t, reviewtext)
encoded_docs<- texts_to_matrix(t, reviewtext, mode='count')
vocab_size = length(t$word_index) + 1

maxlen=180
data<-pad_sequences(texts_sequence, maxlen=maxlen)
labels <- as.numeric(dt$top)

train_ind <- sample(seq_len(nrow(dt)), size = 1500)


train <- data[train_ind, ]
test <- data[-train_ind, ]

train_labels <- labels[train_ind]
test_labels <- labels[-train_ind ]

model <- keras_model_sequential()

model %>% 
  layer_embedding(input_dim =20000, 
                  output_dim = 508,  input_length = maxlen) %>%
  layer_dropout(0.1) %>%
  layer_conv_1d(filters=50, kernel_size=5, 
                padding = "same", activation = "relu", strides = 1 ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units=500) %>%
    layer_dropout(0.1) %>%
  layer_activation("relu") %>%
  layer_dense(units=1, activation='sigmoid') %>% 
  compile(loss='binary_crossentropy', optimizer='nadam', metrics = c('accuracy'))


history<-model %>%
  fit(
    train, train_labels,
    batch_size = 100,
    epochs = 50,
    validation_data = list(test, test_labels)
  )
##########################################################################
##########################################################################

create_ngram_set <- function(input_list, ngram_value = 2){
  indices <- map(0:(length(input_list) - ngram_value), ~1:ngram_value + .x)
  indices %>%
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique()
}

add_ngram <- function(sequences, token_indice, ngram_range = 2){
  ngrams <- map(
    sequences, 
    create_ngram_set, ngram_value = ngram_range
  )
  
  seqs <- map2(sequences, ngrams, function(x, y){
    tokens <- token_indice$token[token_indice$ngrams %in% y]  
    c(x, tokens)
  })
  
  seqs
}
##########################################################################
##########################################################################
train <- unname(as.list(as.data.frame(t(train))))
test <- unname(as.list(as.data.frame(t(test))))


ngrams <- train %>% 
  map(create_ngram_set) %>%
  unlist() %>%
  unique()

token_indice <- data.frame(
  ngrams = ngrams,
  token  = 1:length(ngrams) + (5000), 
  stringsAsFactors = FALSE  )

max_features <- max(token_indice$token) + 1
train1 <- add_ngram(train, token_indice, 2)
test1 <- add_ngram(test, token_indice, 2)
train1 <- pad_sequences(train1, maxlen = 150)
test1 <- pad_sequences(test1, maxlen = 150)

###################
model <- keras_model_sequential()
#embedding_dims <- 50
maxlen=150
model %>%
  layer_embedding( input_dim =200000,  output_dim = 800,  input_length = maxlen ) %>%
  layer_dropout(0.2) %>%
  layer_dense(units=800) %>%
  layer_dropout(0.1) %>%
  layer_global_average_pooling_1d() %>%
  layer_activation("relu") %>%
  layer_dense(units=1, activation='sigmoid') %>% 
  compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# Fitting -----------------------------------------------------------------

history<-model %>% fit(
  train1, train_labels, 
  batch_size = 80,
  epochs = 20,
  validation_data = list(test1, test_labels)
)

library(ggplot2)
plot(history)+scale_colour_discrete(name="Dataset",
                                    labels=c("Train", "Test"))
