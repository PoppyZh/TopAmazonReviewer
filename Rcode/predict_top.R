
library(readr)
library(keras)
reviewr_sub_samplereview <- read_csv("~/1005Inference/reviewr_sub_samplereview.tsv")




reviewr_sub_sample <- read_csv("~/1005Inference/reviewr_sub_sample.tsv")
## Recoding reviewr_sub_sample$gender into reviewr_sub_sample$gender_rec
reviewr_sub_sample$gender_rec <- recode(reviewr_sub_sample$gender,
               "male" = "-1",
               "female" = "1",
               .missing = "0")

reviewr_sub_samplereview$gender_rec <- recode(reviewr_sub_samplereview$gender,
                                        "male" = -1,
                                        "female" = 1,
                                        .missing = 0)


reviewr_sub_sample$top<- ifelse( reviewr_sub_sample$n_review>100, 1, 0)
top10<-merge(reviewr_sub_samplereview, reviewr_sub_sample, by="userid")


# we subset to the top 10 reviews for each reviewer. If a reviewer has less than 10 review, 
# we just use however many they have 

top10<-reviewr_sub_samplereview%>%group_by(userid)%>%
  arrange(date)%>%mutate(rank=order(date))%>%
  filter(rank<11)


# we can also try with using just 1 review 

top1<-reviewr_sub_samplereview%>%group_by(userid)%>%
  arrange(date)%>%mutate(rank=order(date))%>%
  filter(rank<6)


total<-top1%>%group_by(userid)%>%
  summarise(mean_score= mean(score,na.rm=TRUE), 
            mean_len= mean(len,na.rm=TRUE), 
            mean_score_relative= mean(score_relative,na.rm=TRUE), 
            mean_sequence= mean(review_sequence,na.rm=TRUE), 
            mean_helpful= sum(helpful,na.rm=TRUE)/sum(total,na.rm=TRUE), 
            gender= mean(gender_rec,na.rm=TRUE), 
            mean_totaln= mean(n_review))
total$top<- ifelse( total$mean_totaln>100, 1, 0)
total[is.na(total)] <- 0.5

train_ind <- sample(seq_len(nrow(total)), size = 1400)

train <- total[train_ind, ]
test <- total[-train_ind, ]


X_train<-train%>%select(mean_score, mean_len,mean_sequence,
                   mean_score_relative, mean_helpful, gender)%>%data.matrix(.)
X_train <- scale(X_train)
y_train<-train%>%select(top)%>%data.matrix(.)


X_test<-test%>%select(mean_score, mean_len,mean_sequence,
                        mean_score_relative, mean_helpful, gender)%>%data.matrix(.)
X_test <- scale(X_test)
y_test<-test%>%select(top)%>%data.matrix(.)
################################

# very basic nn
##############################
model <- keras_model_sequential()

# add layers and compile the model
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(6)) %>% 
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )

# Train the model, iterating on the data in batches of 32 samples
history<-model %>% fit(X_train, y_train,  epochs=50, batch_size=32, 
                       validation_split = 0.2)

plot(history)

score <- model %>% evaluate(X_test, y_test, batch_size = 32)
score$acc 

################################

# very basic nn with text elements 
##############################
#data.table::fwrite(text_sample, file="sampelreview.csv")
#devtools::install_github("statsmaths/cleanNLP")
library(text2vec)
library(data.table)

text_sample<-fread("sampelreview.csv")
text_sample
top10$reviewerID<-top10$userid
top10$productID<-top10$productid
top10<-unique(top10)
text_sample<-unique(text_sample)

text<-merge(text_sample, top10, all.y=TRUE, by=c("reviewerID", "productID"))
fwrite(text, "sampelreview_top10.csv")


train_text<-merge(text, train, by="userid", all.y=TRUE)
test_text<-merge(text, test, by="userid", all.y=TRUE)

prep_fun = tolower
tok_fun = word_tokenizer
#all_ids = movie_review$id


it_train = itoken(train_text$reviewText, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train_text$userid, 
                  progressbar = FALSE)

vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
#t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)
identical(rownames(dtm_train), train_text$userid)


library(tidytext)
library(dplyr)
library(janeaustenr)
library(tidyr)

book_words <- austen_books() %>%
  unnest_tokens(word, text) %>%
  count(book, word, sort = TRUE) %>%
  ungroup()

total_words <- book_words %>% 
  group_by(book) %>% 
  summarize(total = sum(n))

book_words <- book_words %>%
  bind_tf_idf(word, book, n)%>%
  select(book, word, tf_idf)%>%
  spread(word, tf_idf)



