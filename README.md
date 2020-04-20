# TopAmazonReviewer

## Introduction 

The success and popularity of Amazon reviews should be credited to the hard-working Amazon reviewers. According to the Amazon Hall-of-Fame, a top reviewer named “iiiireader”, has written as many as 4000 reviews, which accumulated him or her more than 44,339 helpfulness votes by Oct 2017. Amazon recognizes around 10,000 top reviewers like “iiiireader” every month, and encourages reviewers to continue contributing high-quality reviewers to the community. Some of the top reviewers can even write Amazon reviews for living. 

This project aims to investigate how the top reviewers develop their review-paths on Amazon. In particular, we are interested in exploring how reviewers review products from different categories. Amazon started by selling books so top reviewers are very likely to start reviewing books in their early stages. However, as Amazon integrates more and more product categories, some reviewers developed their paths by reviewing products from various categories while some maintained their expertise within one category.  We are interested in exploring Amazon reviewers’ review-paths, and understanding how the review-path choices relate to reviewers characteristics, such as gender. 

## Data
Our data is a random subset of reviewers from a bigger data set that is used in the research project by Qianyun Zhang and Professor Vishal Singh. It includes all the reviews a focal reviewers have written, across all product categories on Amazon. 

## Goal  
Predict whether people will be a top reviewer (with more than 100 reviews) or not using their first 1 review. 

## Model 

1. MLP 
2. 1-d CNN 
3. FastText

|Model          | Numerical Data    | Numerical and Textual Attributes | Review Text |
| ------------- |:-----------------:|:--------------------------------:|-----------:|
| MLP           | 1st Reviews: 61\% | 1st Reviews: 80\%                |       |
|MLP            | 3 Reviews: 71\%   | 3 Reviews: 81\%                  |      |
|MLP            | 10 Reviews: 78\%  | 10 Reviews: 83\%                 |     |
|LTSM           |                   | 10 Reviews: 84\%                 |  |
|1-d CNN        |                   |                                  |10 Reviews: 87\%  |
|Fasttext       |                   |                                  | Reviews: 90\%|


## Findings

(1) Using our model, we can predict who will become top reviewers with 90% accuracy. Review text help to improve prediction rate. 
(2) I found 4 types of top reviewers who have distinct interests in diﬀerent product categories. 
