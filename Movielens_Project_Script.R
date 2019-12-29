#############################################################
# Movielens Project Report
# Date: September 21,2019
# Repo: https://github.com/alexismenanieves/Movielens_Project
#############################################################

# Step 1 - Get the data, understand it and tidy it if necessary ----------------------

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Added code to avoid downloading if data exists

if (!file.exists("ml-10M100K/ratings.dat") || !file.exists("ml-10M100K/movies.dat")) {
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  
} else {
  ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)  
}

colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# A first view of the data, dimensions and variables
dim(edx)
as_tibble(edx)
summary(edx)

# Extract the date from edx and validation sets
edx <- edx %>% 
  mutate(date = as.Date.POSIXct(timestamp)) %>% select(-timestamp)
validation <- validation %>% 
  mutate(date = as.Date.POSIXct(timestamp)) %>% select(-timestamp)

# Use regex to extract the year from the movie title
edx <- edx %>% 
  mutate(year = as.integer(str_extract(str_extract(title,"\\((\\d{4})\\)$"),"\\d{4}")))
validation <- validation %>% 
  mutate(year = as.integer(str_extract(str_extract(title,"\\((\\d{4})\\)$"),"\\d{4}")))

# Step 2 - Exploratory analysis ---------------------------------------------------------

edx %>% summarize(n_movies = n_distinct(movieId), n_users = n_distinct(userId), n_years = n_distinct(year))

edx %>% count(movieId) %>% ggplot(aes(n)) + 
  geom_histogram(bins = 20, color="black") + scale_x_log10() + ggtitle("Fig. 1 - Movies")

edx %>% count(userId) %>% ggplot(aes(n)) + 
  geom_histogram(bins = 20, color="black") + scale_x_log10() + ggtitle("Fig. 2 - Users")

edx %>% group_by(rating) %>% tally()

edx %>% group_by(year) %>% summarize(count=n()/1000) %>% ggplot(aes(year,count)) + 
  geom_line() + ggtitle("Fig. 3 - Movie year ratings") + ylab("Count in thousands")

edx %>% group_by(year) %>% 
  summarize(average = mean(rating)) %>% ggplot(aes(year,average)) + 
  geom_point() + ggtitle("Fig. 4 - Movie year average rating") 

edx %>% group_by(date) %>% 
  summarize(rating = mean(rating)) %>% ggplot(aes(date,rating)) + geom_point() + 
  ggtitle("Fig. 5 - Rating date average")

edx %>% group_by(movieId) %>% summarize(n=n(),year=as.character(first(year))) %>% 
  qplot(year,n,data=.,geom="boxplot", main = "Fig. 6 - Ratings per year Boxplot ") + coord_trans(y="sqrt") + 
  theme(axis.text.x=element_text(angle=90,hjust=1))

# Step 3 - Modelling and results --------------------------------------------------------

#   Firstly we create a training and test set from edx dataset, in order to tune our model
set.seed(1996)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2,list = FALSE)
train_set <- edx[-test_index,]
test_set_tmp <- edx[test_index,]
test_set <- test_set_tmp %>% 
  semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
train_set <- rbind(train_set, anti_join(test_set_tmp, test_set))
rm(test_set_tmp)

#   Sencondly, we apply a basic model on training dataset:
mu <- mean(edx$rating)
mu
lambdas <- seq(0,10,0.25)

# And a basic RMSE function to evaluate
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#   3.1 Naive approach
mu <- mean(edx$rating)
naive_rmse <- RMSE(test_set$rating,mu)
rmse_results <- data_frame(Method = "Just the average", RMSE = naive_rmse)

#   3.2 Movie effect
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  pull(b_i)
movie_effect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Movie effect", 
                                     RMSE = movie_effect_rmse))

#   3.3 User effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)
user_effect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "User effect", 
                                     RMSE = user_effect_rmse))
#   3.4 Year effect
year_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  group_by(year) %>% 
  summarize(b_y = mean(rating - mu - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  left_join(year_avgs, by = "year") %>% 
  mutate(pred = mu + b_i + b_u + b_y) %>% 
  pull(pred)
RMSE(predicted_ratings, test_set$rating)

#   3.4 Regularized movie + user effect
lambdas <- seq(0,10,0.25)

rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating -  mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Year effect
  b_y <- train_set %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    group_by(year) %>% 
    summarize(b_y = sum(rating - b_u - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>% 
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
  
# Lets make a lambdas vs rmses plot to see where is the best lambda
qplot(lambdas,rmses)
best_lambda <- lambdas[which.min(rmses)]
best_lambda
min(rmses)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "Regularized movie + user + year effect", 
                                     RMSE = min(rmses)))

# Test the result against validation set

b_i_tuned <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i_tuned = sum(rating - mu)/(n() + best_lambda))
b_u_tuned <- edx %>% 
  left_join(b_i_tuned, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u_tuned = sum(rating - b_i_tuned - mu)/(n() + best_lambda))
b_y_tuned <- edx %>% 
  left_join(b_i_tuned, by = "movieId") %>% 
  left_join(b_u_tuned, by = "userId") %>% 
  group_by(year) %>% 
  summarize(b_y_tuned = sum(rating - b_i_tuned - b_u_tuned - mu)/(n() + best_lambda))
predicted_ratings <- validation %>% 
  left_join(b_i_tuned, by = "movieId") %>% 
  left_join(b_u_tuned, by = "userId") %>% 
  left_join(b_y_tuned, by = "year") %>% 
  mutate(pred = mu + b_i_tuned + b_u_tuned + b_y_tuned) %>% 
  pull(pred)
final_rmse <- RMSE(predicted_ratings, validation$rating)
final_rmse

