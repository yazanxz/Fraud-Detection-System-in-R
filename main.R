# Fraud Detection in Financial Transactions
# Using Multiple Anomaly Detection Methods: Isolation Forest, LOF, and Autoencoders

# Required libraries
library(dplyr)        # For data manipulation
library(ggplot2)      # For visualization
library(randomForest) # For isolation forest implementation
library(dbscan)       # For LOF
library(keras)        # For autoencoder
library(lubridate)    # For date/time handling
library(caret)        # For data preprocessing
library(scales)       # For formatting output

# ---- 1. Data Generation ----
# Let's create synthetic transaction data
set.seed(123)

# Number of transactions
n_transactions <- 10000
n_fraud <- 100

# Generating normal transactions
normal_transactions <- data.frame(
  transaction_id = seq(1, n_transactions - n_fraud),
  timestamp = sample(seq(as.POSIXct('2024-01-01'), as.POSIXct('2024-03-31'), by="hour"), 
                     n_transactions - n_fraud, replace=TRUE),
  amount = rlnorm(n_transactions - n_fraud, meanlog=4, sdlog=1), # Log-normal for realistic amounts
  location_distance = rnorm(n_transactions - n_fraud, mean=10, sd=5), # Distance from usual locations
  day_since_last_transaction = rpois(n_transactions - n_fraud, lambda=3) + 1, # Time pattern
  merchant_category = sample(1:20, n_transactions - n_fraud, replace=TRUE),
  is_fraud = 0
)

# Generating fraudulent transactions
fraud_transactions <- data.frame(
  transaction_id = seq(n_transactions - n_fraud + 1, n_transactions),
  timestamp = sample(seq(as.POSIXct('2024-01-01'), as.POSIXct('2024-03-31'), by="hour"), 
                    n_fraud, replace=TRUE),
  # Higher amounts for fraud
  amount = rlnorm(n_fraud, meanlog=6.5, sdlog=1.5),
  # Unusual locations (higher distance)
  location_distance = rnorm(n_fraud, mean=50, sd=15),
  # Unusual timing
  day_since_last_transaction = rpois(n_fraud, lambda=0.5) + 1,
  # Unusual merchant categories
  merchant_category = sample(c(21:25, sample(1:20, 10)), n_fraud, replace=TRUE),
  is_fraud = 1
)

# Combine datasets
transactions <- rbind(normal_transactions, fraud_transactions) %>%
  arrange(timestamp)

# Add time-based features
transactions <- transactions %>%
  mutate(
    hour_of_day = hour(timestamp),
    day_of_week = wday(timestamp),
    is_weekend = ifelse(day_of_week %in% c(1, 7), 1, 0),
    is_night = ifelse(hour_of_day >= 22 | hour_of_day <= 5, 1, 0)
  )

# ---- 2. Exploratory Data Analysis ----
# Visualize transaction amounts
ggplot(transactions, aes(x=amount, fill=factor(is_fraud))) +
  geom_histogram(bins=50, alpha=0.7) +
  scale_x_log10(labels=dollar_format()) +
  labs(title="Distribution of Transaction Amounts",
       x="Amount (log scale)",
       y="Count",
       fill="Fraud") +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), labels=c("Normal", "Fraud"))

# Visualize transaction timing patterns
ggplot(transactions, aes(x=hour_of_day, fill=factor(is_fraud))) +
  geom_density(alpha=0.7) +
  labs(title="Transaction Timing Distribution",
       x="Hour of Day",
       y="Density",
       fill="Fraud") +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), labels=c("Normal", "Fraud"))

# Location distance vs amount
ggplot(transactions, aes(x=location_distance, y=amount, color=factor(is_fraud))) +
  geom_point(alpha=0.5) +
  scale_y_log10(labels=dollar_format()) +
  labs(title="Transaction Amount vs. Location Distance",
       x="Distance from Usual Locations",
       y="Amount (log scale)",
       color="Fraud") +
  theme_minimal() +
  scale_color_manual(values=c("blue", "red"), labels=c("Normal", "Fraud"))

# ---- 3. Data Preprocessing ----
# Scale the features for anomaly detection algorithms
preprocess_features <- c("amount", "location_distance", "day_since_last_transaction", 
                         "hour_of_day", "day_of_week", "is_weekend", "is_night", "merchant_category")

# Create preprocessing model
preproc <- preProcess(transactions[, preprocess_features], method=c("center", "scale"))
transactions_scaled <- predict(preproc, transactions[, preprocess_features])

# ---- 4. Anomaly Detection Models ----

# 4.1 Isolation Forest
# Using randomForest as an approximate implementation of Isolation Forest in R
isolationForest <- function(data, ntrees=100, sample_size=256) {
  results <- rep(0, nrow(data))
  
  for (i in 1:ntrees) {
    # Sample rows and columns
    rows <- sample(nrow(data), min(sample_size, nrow(data)))
    cols <- sample(ncol(data), max(1, floor(sqrt(ncol(data)))))
    
    # Build a tree
    tree <- randomForest(x=data[rows, cols, drop=FALSE], 
                         y=factor(rep(1, length(rows))),
                         ntree=1, 
                         mtry=length(cols))
    
    # Measure tree depth for each point (higher depth = less anomalous)
    tree_depth <- predict(tree, data, nodes=TRUE)
    
    # Convert depth to anomaly score (normalize)
    max_depth <- max(tree_depth)
    results <- results + (max_depth - tree_depth) / max_depth
  }
  
  # Average scores across trees
  results / ntrees
}

# Apply Isolation Forest
isolation_forest_scores <- isolationForest(transactions_scaled)
transactions$isolation_forest_score <- isolation_forest_scores

# 4.2 Local Outlier Factor (LOF)
lof_result <- lof(transactions_scaled, k=10)
transactions$lof_score <- lof_result

# 4.3 Autoencoder
# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units=64, activation="relu", input_shape=ncol(transactions_scaled)) %>%
  layer_dense(units=32, activation="relu") %>%
  layer_dense(units=16, activation="relu") %>%
  layer_dense(units=8, activation="relu") %>%
  layer_dense(units=16, activation="relu") %>%
  layer_dense(units=32, activation="relu") %>%
  layer_dense(units=64, activation="relu") %>%
  layer_dense(units=ncol(transactions_scaled))

# Compile the model
model %>% compile(
  optimizer="adam",
  loss="mse"
)

# Train the autoencoder
history <- model %>% fit(
  x=as.matrix(transactions_scaled),
  y=as.matrix(transactions_scaled),
  epochs=50,
  batch_size=32,
  validation_split=0.2,
  verbose=0
)

# Get reconstruction error as anomaly score
reconstructed <- model %>% predict(as.matrix(transactions_scaled))
reconstruction_error <- apply(as.matrix(transactions_scaled) - reconstructed, 1, function(x) sum(x^2))
transactions$autoencoder_score <- reconstruction_error

# ---- 5. Anomaly Score Combination and Evaluation ----

# Normalize all scores to [0,1]
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

transactions$isolation_forest_score_norm <- normalize(transactions$isolation_forest_score)
transactions$lof_score_norm <- normalize(transactions$lof_score)
transactions$autoencoder_score_norm <- normalize(transactions$autoencoder_score)

# Combined anomaly score (average of normalized scores)
transactions$combined_score <- (transactions$isolation_forest_score_norm + 
                                transactions$lof_score_norm + 
                                transactions$autoencoder_score_norm) / 3

# Sort by combined score to find most anomalous transactions
suspicious_txns <- transactions %>%
  arrange(desc(combined_score)) %>%
  select(transaction_id, timestamp, amount, location_distance, 
         combined_score, is_fraud)

# Evaluate model performance
evaluate_model <- function(scores, labels, top_n=NULL) {
  # If top_n is provided, only consider top_n transactions as positives
  if (!is.null(top_n)) {
    predicted <- rep(0, length(scores))
    predicted[order(scores, decreasing=TRUE)[1:top_n]] <- 1
  } else {
    # Determine optimal threshold using ROC
    roc_data <- data.frame(
      score = scores,
      label = labels
    )
    
    # Create various thresholds and calculate metrics
    thresholds <- seq(min(scores), max(scores), length.out=100)
    best_f1 <- 0
    best_threshold <- 0
    
    for (threshold in thresholds) {
      predicted <- ifelse(scores >= threshold, 1, 0)
      tp <- sum(predicted == 1 & labels == 1)
      fp <- sum(predicted == 1 & labels == 0)
      fn <- sum(predicted == 0 & labels == 1)
      
      precision <- if (tp + fp > 0) tp / (tp + fp) else 0
      recall <- if (tp + fn > 0) tp / (tp + fn) else 0
      f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
      
      if (f1 > best_f1) {
        best_f1 <- f1
        best_threshold <- threshold
      }
    }
    
    predicted <- ifelse(scores >= best_threshold, 1, 0)
  }
  
  # Calculate metrics
  tp <- sum(predicted == 1 & labels == 1)
  fp <- sum(predicted == 1 & labels == 0)
  tn <- sum(predicted == 0 & labels == 0)
  fn <- sum(predicted == 0 & labels == 1)
  
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  
  return(list(
    precision = precision,
    recall = recall,
    f1 = f1,
    accuracy = accuracy,
    detected_fraud = tp,
    total_fraud = tp + fn
  ))
}

# Evaluate each method
methods <- list(
  "Isolation Forest" = transactions$isolation_forest_score_norm,
  "LOF" = transactions$lof_score_norm,
  "Autoencoder" = transactions$autoencoder_score_norm,
  "Combined" = transactions$combined_score
)

# Flagging top 5% as suspicious
suspected_count <- ceiling(nrow(transactions) * 0.05)

evaluation_results <- data.frame()
for (method_name in names(methods)) {
  result <- evaluate_model(methods[[method_name]], transactions$is_fraud, suspected_count)
  evaluation_results <- rbind(evaluation_results, 
                             data.frame(
                               Method = method_name,
                               Precision = result$precision,
                               Recall = result$recall,
                               F1_Score = result$f1,
                               Detected_Fraud = result$detected_fraud,
                               Total_Fraud = result$total_fraud
                             ))
}

print(evaluation_results)

# ---- 6. Visualization of Results ----

# Create a scatter plot of all transactions colored by anomaly score
ggplot(transactions, aes(x=location_distance, y=amount, color=combined_score)) +
  geom_point(alpha=0.7) +
  scale_y_log10(labels=dollar_format()) +
  scale_color_gradient(low="blue", high="red") +
  labs(title="Transaction Anomaly Detection Results",
       subtitle="Transactions colored by combined anomaly score",
       x="Distance from Usual Locations",
       y="Amount (log scale)",
       color="Anomaly Score") +
  theme_minimal()

# Highlight actual fraud vs detected anomalies
transactions$detection_status <- case_when(
  transactions$is_fraud == 1 & transactions$combined_score >= quantile(transactions$combined_score, 0.95) ~ "True Positive",
  transactions$is_fraud == 0 & transactions$combined_score >= quantile(transactions$combined_score, 0.95) ~ "False Positive",
  transactions$is_fraud == 1 & transactions$combined_score < quantile(transactions$combined_score, 0.95) ~ "False Negative",
  TRUE ~ "True Negative"
)

ggplot(transactions, aes(x=location_distance, y=amount, color=detection_status)) +
  geom_point(alpha=0.7) +
  scale_y_log10(labels=dollar_format()) +
  labs(title="Fraud Detection Results",
       subtitle="Comparison of actual fraud vs. detected anomalies",
       x="Distance from Usual Locations",
       y="Amount (log scale)",
       color="Status") +
  theme_minimal() +
  scale_color_manual(values=c("red", "orange", "green", "blue"))

# ---- 7. Production-Ready Fraud Detection Function ----

# Create a function that can be used on new transaction data
detect_fraud <- function(new_transactions, model, preproc, threshold=0.95) {
  # Preprocess the new transactions
  scaled_txns <- predict(preproc, new_transactions[, preprocess_features])
  
  # Apply the three anomaly detection methods
  iso_score <- isolationForest(scaled_txns)
  lof_score <- lof(scaled_txns, k=10)
  
  # Autoencoder
  reconstructed <- model %>% predict(as.matrix(scaled_txns))
  ae_score <- apply(as.matrix(scaled_txns) - reconstructed, 1, function(x) sum(x^2))
  
  # Normalize scores
  iso_score_norm <- (iso_score - min(transactions$isolation_forest_score)) / 
                    (max(transactions$isolation_forest_score) - min(transactions$isolation_forest_score))
  lof_score_norm <- (lof_score - min(transactions$lof_score)) / 
                    (max(transactions$lof_score) - min(transactions$lof_score))
  ae_score_norm <- (ae_score - min(transactions$autoencoder_score)) / 
                   (max(transactions$autoencoder_score) - min(transactions$autoencoder_score))
  
  # Combine scores
  combined_score <- (iso_score_norm + lof_score_norm + ae_score_norm) / 3
  
  # Flag suspicious transactions
  is_suspicious <- combined_score > quantile(transactions$combined_score, threshold)
  
  # Return results
  return(data.frame(
    new_transactions,
    isolation_forest_score = iso_score_norm,
    lof_score = lof_score_norm,
    autoencoder_score = ae_score_norm,
    combined_score = combined_score,
    is_suspicious = is_suspicious
  ))
}

# ---- 8. Sample Usage with New Data ----

# Generate a few new transactions (some normal, some suspicious)
new_txns <- data.frame(
  transaction_id = seq(1, 10),
  timestamp = sample(seq(as.POSIXct('2024-04-01'), as.POSIXct('2024-04-02'), by="hour"), 10),
  amount = c(rlnorm(8, meanlog=4, sdlog=1), rlnorm(2, meanlog=7, sdlog=1)),
  location_distance = c(rnorm(8, mean=10, sd=5), rnorm(2, mean=60, sd=10)),
  day_since_last_transaction = c(rpois(8, lambda=3) + 1, rpois(2, lambda=0.2) + 1),
  merchant_category = c(sample(1:20, 8, replace=TRUE), 22, 24)
)

# Add time-based features
new_txns <- new_txns %>%
  mutate(
    hour_of_day = hour(timestamp),
    day_of_week = wday(timestamp),
    is_weekend = ifelse(day_of_week %in% c(1, 7), 1, 0),
    is_night = ifelse(hour_of_day >= 22 | hour_of_day <= 5, 1, 0)
  )

# Apply fraud detection
results <- detect_fraud(new_txns, model, preproc, threshold=0.95)

# Show suspicious transactions
suspicious_results <- results %>%
  filter(is_suspicious) %>%
  select(transaction_id, timestamp, amount, location_distance, 
         combined_score, is_suspicious)

print("Suspicious Transactions Detected:")
print(suspicious_results)

# ---- 9. Interpretable Results ----

# Create a function to explain why a transaction is suspicious
explain_suspicious_transaction <- function(txn, threshold_percentiles=list(
  amount=0.95, location_distance=0.95, day_since_last_transaction=0.05,
  is_night=NA, is_weekend=NA)) {
  
  reasons <- c()
  
  # Check if amount is unusually high
  amount_threshold <- quantile(transactions$amount, threshold_percentiles$amount)
  if (!is.na(threshold_percentiles$amount) && txn$amount > amount_threshold) {
    reasons <- c(reasons, paste0("Unusually high amount: $", round(txn$amount, 2), 
                                " (above ", round(amount_threshold, 2), ")"))
  }
  
  # Check if location is unusual
  dist_threshold <- quantile(transactions$location_distance, threshold_percentiles$location_distance)
  if (!is.na(threshold_percentiles$location_distance) && txn$location_distance > dist_threshold) {
    reasons <- c(reasons, paste0("Unusual location: distance score of ", round(txn$location_distance, 2),
                                " (above ", round(dist_threshold, 2), ")"))
  }
  
  # Check if timing is unusual (too soon after last transaction)
  day_threshold <- quantile(transactions$day_since_last_transaction, threshold_percentiles$day_since_last_transaction)
  if (!is.na(threshold_percentiles$day_since_last_transaction) && 
      txn$day_since_last_transaction < day_threshold) {
    reasons <- c(reasons, paste0("Unusual timing: only ", txn$day_since_last_transaction, 
                                " days since last transaction (below ", day_threshold, ")"))
  }
  
  # Check if night transaction
  if (!is.na(threshold_percentiles$is_night) && txn$is_night == 1) {
    reasons <- c(reasons, "Transaction occurred at night (higher risk)")
  }
  
  # Unusual merchant category
  if (txn$merchant_category > 20) {
    reasons <- c(reasons, paste0("Unusual merchant category: ", txn$merchant_category))
  }
  
  # If no specific reasons found, use the anomaly scores
  if (length(reasons) == 0) {
    if (txn$isolation_forest_score > 0.8) {
      reasons <- c(reasons, "Flagged by Isolation Forest algorithm")
    }
    if (txn$lof_score > 0.8) {
      reasons <- c(reasons, "Flagged by Local Outlier Factor algorithm")
    }
    if (txn$autoencoder_score > 0.8) {
      reasons <- c(reasons, "Unusual pattern detected by Autoencoder")
    }
  }
  
  # Overall score
  reasons <- c(reasons, paste0("Combined anomaly score: ", round(txn$combined_score, 3)))
  
  return(reasons)
}

# Apply explanation to suspicious transactions
if (nrow(suspicious_results) > 0) {
  for (i in 1:nrow(suspicious_results)) {
    txn_id <- suspicious_results$transaction_id[i]
    txn <- results[results$transaction_id == txn_id, ]
    cat("\nTransaction ID:", txn_id, "\n")
    cat("Reasons for flagging:\n")
    reasons <- explain_suspicious_transaction(txn)
    cat(paste0("- ", reasons, collapse="\n"), "\n")
  }
}

# ---- 10. Save the model for future use ----
saveRDS(list(
  preproc = preproc,
  model = model,
  thresholds = list(
    combined_score = quantile(transactions$combined_score, 0.95)
  )
), "fraud_detection_model.rds")

# Usage instructions
cat("\n--- Fraud Detection System Ready ---\n")
cat("To use this system on new transaction data:\n")
cat("1. Load the saved model: model <- readRDS('fraud_detection_model.rds')\n")
cat("2. Format your new transaction data with the required features\n")
cat("3. Detect fraud: results <- detect_fraud(new_transactions, model$model, model$preproc)\n")
cat("4. Review flagged transactions: filter(results, is_suspicious)\n")
