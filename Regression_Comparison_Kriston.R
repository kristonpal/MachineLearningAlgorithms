#Installing the required packages
  install.packages('e1071')
  install.packages('rpart')
  install.packages('randomForest')
  install.packages('Metrics')
  install.packages('caTools')
  install.packages('reshape2')

#Using the required packages
  library(e1071)
  library(rpart)
  library(randomForest)
  library(Metrics)
  library(caTools)
  library(ggplot2)
  library(reshape2)
  library(gridExtra)

# Loading the dataset
  data <- read.csv("altered_white_wine_quality_dataset_22.csv")

  
##### Data Preprocessing
  
  # Encoding categorical data
  data$type = factor(data$type,
                         levels = c('white', 'red'),
                         labels = c(1, 2))
  
# Dealing with the missing values
  # Replacing NA values with the mean for each column
  my_data <- lapply(data, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
  # Converting the list back to a data frame
  my_data <- as.data.frame(my_data)
  
# Splitting the dataset into training and testing sets
  set.seed(123)
  split = sample.split(my_data$quality, SplitRatio = 0.8) # 80% for training & 20% for validation
  training_set = subset(my_data, split == TRUE)
  test_set = subset(my_data, split == FALSE)


# Feature Scaling of Independent variables
# Selecting only the numeric columns for scaling
  numeric_cols <- sapply(training_set, is.numeric)
  training_data_scaled <- training_set
  testing_data_scaled <- test_set
  training_data_scaled[, numeric_cols] <- scale(training_data_scaled[, numeric_cols])
  testing_data_scaled[, numeric_cols] <- scale(testing_data_scaled[, numeric_cols])

  
##### Building the models

# Linear Regression
  lm_model <- lm(quality ~ ., data = training_data_scaled)
  lm_predictions <- predict(lm_model, newdata = testing_data_scaled)
  summary(lm_model)#summarizing the regression result
  
  #Calculating MAE and RMSE values 
  lm_mae <- mae(testing_data_scaled$quality, lm_predictions)
  lm_rmse <- rmse(testing_data_scaled$quality, lm_predictions)


# Support Vector Regression (SVR)
  svr_model <- svm(quality ~ ., data = training_data_scaled)
  svr_predictions <- predict(svr_model, testing_data_scaled)
  
  #Calculating MAE and RMSE values
  svr_mae <- mae(testing_data_scaled$quality, svr_predictions)
  svr_rmse <- rmse(testing_data_scaled$quality, svr_predictions)

# Polynomial Regression
  
  # Polynomial Regression for "quality" using selected independent variables
  poly_model <- lm(quality ~ poly(alcohol, 2) + fixed.acidity + volatile.acidity + citric.acid + 
                     residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                     density + pH + sulphates + type, data = training_data_scaled)
  poly_predictions <- predict(poly_model, newdata = testing_data_scaled)
  
  #Calculating MAE and RMSE values
  poly_mae <- mae(testing_data_scaled$quality, poly_predictions)
  poly_rmse <- rmse(testing_data_scaled$quality, poly_predictions)

# Decision Tree Regression
  d_tree_model <- rpart(quality ~ ., data = training_data_scaled)
  d_tree_predictions <- predict(d_tree_model, testing_data_scaled)
  
  #Calculating MAE and RMSE values
  d_tree_mae <- mae(testing_data_scaled$quality, d_tree_predictions)
  d_tree_rmse <- rmse(testing_data_scaled$quality, d_tree_predictions)

# Random Forest Regression
  rf_model <- randomForest(quality ~ ., data = training_data_scaled, ntree = 800)  # Adjusted the ntree value to 800 comparing
                                                                                # to various values like 300,500, and 1000.
  rf_predictions <- predict(rf_model, testing_data_scaled)
  
  #Calculating MAE and RMSE values
  rf_mae <- mae(testing_data_scaled$quality, rf_predictions)
  rf_rmse <- rmse(testing_data_scaled$quality, rf_predictions)

  
##### Evaluating the models
  
# Compare performance using MAE and RMSE
  results <- data.frame(
    Model = c("Linear Regression", "SVR", "Polynomial Regression", "Decision Tree", "Random Forest"),
    MAE = c(lm_mae, svr_mae, poly_mae, d_tree_mae, rf_mae),
    RMSE = c(lm_rmse, svr_rmse, poly_rmse, d_tree_rmse, rf_rmse)
  )

  print("Performance Metrics (MAE and RMSE):")
  print(results)

# Creating visualizations to show the MAE and RMSE values
  
  # Melting the data frame for visualization
  melted_results <- melt(results, id.vars = "Model")
  
  # Create a bar chart to compare MAE and RMSE
  ggplot(data = melted_results, aes(x = Model, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Model Performance Comparison",
         y = "Error",
         x = "Model") +
    scale_fill_manual(values = c("MAE" = "purple", "RMSE" = "green")) +
    theme_minimal() +
    theme(legend.title = element_blank(),
          axis.text.x = element_text(angle = 15, hjust = 1))

  
  
  # Creating visualizations for Performance Evaluation of models
  
  
  # Linear Regression Scatterplot
  linear_scatterplot <- ggplot(data = testing_data_scaled, aes(x = quality, y = lm_predictions)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "Linear Regression Scatterplot",
         x = "Actual Quality",
         y = "Predicted Quality")
  
  # SVR Scatterplot
  svr_scatterplot <- ggplot(data = testing_data_scaled, aes(x = quality, y = svr_predictions)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "SVR Scatterplot",
         x = "Actual Quality",
         y = "Predicted Quality")
  
  # Polynomial Regression Scatterplot
  poly_scatterplot <- ggplot(data = testing_data_scaled, aes(x = quality, y = poly_predictions)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "Polynomial Regression Scatterplot",
         x = "Actual Quality",
         y = "Predicted Quality")
  
  # Decision Tree Scatterplot
  d_tree_scatterplot <- ggplot(data = testing_data_scaled, aes(x = quality, y = d_tree_predictions)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "Decision Tree Scatterplot",
         x = "Actual Quality",
         y = "Predicted Quality")
  
  # Random Forest Scatterplot
  rf_scatterplot <- ggplot(data = testing_data_scaled, aes(x = quality, y = rf_predictions)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "Random Forest Scatterplot",
         x = "Actual Quality",
         y = "Predicted Quality")
  
  # Arranging the scatterplots in a grid
  grid.arrange(linear_scatterplot, svr_scatterplot, poly_scatterplot,
               d_tree_scatterplot, rf_scatterplot, ncol = 2)

