#' generate_data: Generate synthetic primary, auxiliary, and noisy datasets for transfer learning
#'
#' This function generates synthetic datasets for the primary task (target domain), auxiliary datasets 
#' (source domains), and noisy datasets for use in transfer learning simulations. It allows flexible input 
#' for the sizes of the auxiliary and noisy datasets, supports different covariance structures, and can 
#' optionally generate test datasets. Users can specify true coefficients or rely on random generation. The function
#' supports generating datasets for both regression and binary classification tasks.
#'
#' @param n0       An integer specifying the number of observations in the primary dataset (target domain).
#' @param p        An integer specifying the dimension, namely the number of predictors. All the generated data must have the same dimension.
#' @param K        An integer specifying the number of auxiliary datasets (source domains).
#' @param nk       Either an integer specifying the number of observations in each auxiliary dataset (source domains),
#'                 or a vector where each element specifies the size of the corresponding auxiliary dataset. If `nk` is 
#'                 a vector, its length must match the number of auxiliary datasets (`K`).
#' @param is_noise Logical; if TRUE, includes noisy data. If FALSE, `K_noise` and `nk_noise` are ignored. Default is TRUE.
#' @param K_noise  An integer specifying the number of noisy auxiliary datasets. If `K_noise = 0`, noisy datasets are skipped.
#'                 If `is_noise = FALSE`, this argument is not used.
#' @param nk_noise Either an integer specifying the number of observations in each noisy dataset, or a vector where 
#'                 each element specifies the size of the corresponding noisy dataset. If `nk_noise` is a vector, its 
#'                 length must match the number of noisy datasets (`K_noise`).
#' @param mu_trgt  A numeric value specifying the mean of the true coefficients in the primary dataset.
#' @param xi_aux   A numeric value representing the shift applied to the true coefficients in the auxiliary datasets.
#' @param ro       A numeric value representing the correlation between predictors (applies to the covariance matrix).
#' @param err_sig  A numeric value specifying the standard deviation of the noise added to the response.
#' @param true_beta A vector of true coefficients for the primary dataset. If `NULL`, it is randomly generated. Default is `NULL`.
#' @param noise_beta A vector of noise coefficients. If `NULL`, it is set to `-true_beta`. Default is `NULL`.
#' @param Sigma_type A string specifying the covariance structure for the predictors. Options are:
#'   "AR" (auto-regressive structure) or "CS" (compound symmetry structure). Default is "AR".
#' @param is_test  Logical; if TRUE, generates test dataset (`X_test`, `y_test`). Default is TRUE.
#' @param n_test   An integer specifying the number of observations in the test data. Default is n0.
#' @param task     A string specifying the type of task. Options are "regression" or "classification". Default is "regression".
#'
#' @return
#' A list containing:
#' \item{X}{The primary dataset predictors (target domain).}
#' \item{y}{The primary dataset responses (target domain).}
#' \item{X_aux}{A list of matrices combining auxiliary and noisy dataset predictors.}
#' \item{y_aux}{A list of vectors combining auxiliary and noisy dataset responses.}
#' \item{X_test}{The test dataset predictors, if `is_test=TRUE`.}
#' \item{y_test}{The test dataset responses, if `is_test=TRUE`.}
#'
#' @details
#' The function first generates a covariance matrix based on the specified `Sigma_type`, then creates the primary dataset 
#' (`X`, `y`), the auxiliary datasets (`X_aux`, `y_aux`), and optionally generates test datasets (`X_test`, `y_test`).
#' The auxiliary datasets are combined with noisy datasets into `X_aux` and `y_aux` for transfer learning use.
#' 
#' If `is_noise = FALSE`, then no noisy data is generated and `K_noise` and `nk_noise` are ignored. If `K_noise = 0`, 
#' noisy data is skipped regardless of the value of `is_noise`. The task can be either "regression" or "classification". 
#' In classification mode, binary response variables are generated using a logistic function.
#'
#' If `nk` or `nk_noise` is a vector, it checks if its length matches the number of auxiliary or noisy +, respectively. 
#' If the lengths do not match, an error is returned.
#'
#' @examples
#' # Example: Generate data with auxiliary, noisy, and test datasets for regression
#' dat_reg <- generate_data(n0=100, p=10, K=3, nk=50, is_noise=TRUE, K_noise=2, nk_noise=30, 
#'                          mu_trgt=1, xi_aux=0.5, ro=0.3, err_sig=1, 
#'                          is_test=TRUE, task="regression")
#' 
#' # Example: Generate data with auxiliary, noisy, and test datasets for classification
#' dat_class <- generate_data(n0=100, p=10, K=3, nk=50, is_noise=TRUE, K_noise=2, nk_noise=30, 
#'                            mu_trgt=1, xi_aux=0.5, ro=0.3, err_sig=1, 
#'                            is_test=TRUE, task="classification")
#' 
#' # Display the dimensions of the generated data
#' cat("Primary dataset (X):", dim(dat_reg$X), "\n")   # Should print 100 x 10 for regression
#' cat("Primary dataset (y):", length(dat_reg$y), "\n") # Should print length 100 for regression
#' 
#' # Display the dimensions of auxiliary datasets
#' cat("Auxiliary dataset 1 (X_aux[[1]]):", dim(dat_reg$X_aux[[1]]), "\n") # Should print 50 x 10
#' cat("Auxiliary dataset 2 (X_aux[[2]]):", dim(dat_reg$X_aux[[2]]), "\n") # Should print 50 x 10
#' 
#' # Display the dimensions of noisy datasets (if generated)
#' cat("Noisy dataset 1 (X_aux[[4]]):", dim(dat_reg$X_aux[[4]]), "\n") # Should print 30 x 10
#' 
#' # Display test data dimensions (if generated)
#' if (!is.null(dat_reg$X_test)) {
#'   cat("Test dataset (X_test):", dim(dat_reg$X_test), "\n") # Should print 100 x 10
#'   cat("Test dataset (y_test):", length(dat_reg$y_test), "\n") # Should print length 100
#' }
#' @export
generate_data <- function(n0, p, K, nk, is_noise=TRUE, K_noise=2, nk_noise=30, 
                         mu_trgt, xi_aux, ro, err_sig, true_beta=NULL, noise_beta=NULL, 
                         Sigma_type="AR", is_test=TRUE, n_test=n0, task="regression") {
  
  # Covariance matrix setup
  if (Sigma_type == "AR") {
    Sigma_mat <- ro ^ abs(outer(seq(p), seq(p), "-"))
  } else if (Sigma_type == "CS") {
    Sigma_mat <- (1 - ro) * diag(p) + ro * matrix(1, p, p)
  }
  
  eig <- eigen(Sigma_mat)
  Sigma_sqrt <- eig$vectors %*% tcrossprod(diag(sqrt(eig$values)), eig$vectors)
  
  # Generate true_beta if not provided
  if (is.null(true_beta)) {
    true_beta <- rnorm(p, mu_trgt, 1)
  } else if (!is.numeric(true_beta) || length(true_beta) != p) {
    stop("`true_beta` must be a numeric vector of length ", p)
  }
  
  # Generate primary data
  X <- matrix(rnorm(n0 * p), n0, p) %*% Sigma_sqrt
  if (task == "regression") {
    y <- X %*% true_beta + rnorm(n0, 0, err_sig)
  } else if (task == "classification") {
    z <- X %*% true_beta + rnorm(n0, 0, err_sig)
    pb <- 1 / (1 + exp(-z))
    y <- rbinom(n0, 1, pb)
  } else {
    stop("Invalid task specified. Choose either 'regression' or 'classification'.")
  }
  
  # Handle auxiliary dataset sizes
  if (length(nk) == 1) {
    nk_vec <- rep(nk, K)
  } else if (length(nk) == K) {
    nk_vec <- nk
  } else {
    stop("The length of 'nk' must be either 1 or match the number of auxiliary datasets 'K'.")
  }
  
  # Handle noisy dataset sizes (if is_noise is TRUE)
  if (!is_noise) {
    K_noise <- 0
    nk_noise <- NULL
  } else if (K_noise == 0) {
    warning("`K_noise = 0` and `is_noise = TRUE` results in no noisy data being generated.")
  }
  
  if (is_noise && K_noise > 0) {
    if (length(nk_noise) == 1) {
      nk_noise_vec <- rep(nk_noise, K_noise)
    } else if (length(nk_noise) == K_noise) {
      nk_noise_vec <- nk_noise
    } else {
      stop("The length of 'nk_noise' must be either 1 or match the number of noisy datasets 'K_noise'.")
    }
  }
  
  # Generate auxiliary data (X_k, y_k)
  true_beta_k <- lapply(seq(K), function(k) true_beta + xi_aux)
  X_k <- lapply(seq(K), function(k) matrix(rnorm(nk_vec[k] * p), nk_vec[k], p) %*% Sigma_sqrt)
  
  if (task == "regression") {
    y_k <- lapply(seq(K), function(k) X_k[[k]] %*% true_beta_k[[k]] + rnorm(nk_vec[k], 0, err_sig))
  } else if (task == "classification") {
    y_k <- lapply(seq(K), function(k) {
      z_k <- X_k[[k]] %*% true_beta_k[[k]] + rnorm(nk_vec[k], 0, err_sig)
      pb_k <- 1 / (1 + exp(-z_k))
      rbinom(nk_vec[k], 1, pb_k)
    })
  }
  
  # Generate noise_beta if not provided
  if (is.null(noise_beta)) {
    noise_beta <- -true_beta
  } else if (!is.numeric(noise_beta) || length(noise_beta) != p) {
    stop("`noise_beta` must be a numeric vector of length ", p)
  }
  
  # Generate noisy data (if is_noise is TRUE and K_noise > 0)
  if (is_noise && K_noise > 0) {
    noise_beta_k <- lapply(seq(K_noise), function(nk) noise_beta - xi_aux)
    X_noise_k <- lapply(seq(K_noise), function(kn) matrix(rnorm(nk_noise_vec[kn] * p), nk_noise_vec[kn], p) %*% Sigma_sqrt)
    
    if (task == "regression") {
      y_noise_k <- lapply(seq(K_noise), function(kn) X_noise_k[[kn]] %*% noise_beta_k[[kn]] + rnorm(nk_noise_vec[kn], 0, err_sig))
    } else if (task == "classification") {
      y_noise_k <- lapply(seq(K_noise), function(kn) {
        z_noise_k <- X_noise_k[[kn]] %*% noise_beta_k[[kn]] + rnorm(nk_noise_vec[kn], 0, err_sig)
        pb_noise_k <- 1 / (1 + exp(-z_noise_k))
        rbinom(nk_noise_vec[kn], 1, pb_noise_k)
      })
    }
    
    # Combine auxiliary and noisy data into X_aux and y_aux
    X_aux <- c(X_k, X_noise_k)
    y_aux <- c(y_k, y_noise_k)
  } else {
    X_aux <- X_k
    y_aux <- y_k
  }
  
  # Initialize return list
  ret <- list(X = X, y = y, X_aux = X_aux, y_aux = y_aux)
  
  # Generate test dataset if required
  if (is_test) {
    X_test <- matrix(rnorm(n_test * p), n_test, p) %*% Sigma_sqrt
    if (task == "regression") {
      y_test <- X_test %*% true_beta + rnorm(n_test, 0, err_sig)
    } else if (task == "classification") {
      z_test <- X_test %*% true_beta + rnorm(n_test, 0, err_sig)
      pb_test <- 1 / (1 + exp(-z_test))
      y_test <- rbinom(n_test, 1, pb_test)
    }
    ret$X_test <- X_test
    ret$y_test <- y_test
  }
  
  return(ret)
}

#' stan: Standardize Training, Validation, and Test Datasets
#'
#' This function standardizes the training, validation, and test datasets by centering and scaling 
#' them using the mean and standard deviation from the training set. It ensures that the validation 
#' and test sets are transformed using the same parameters derived from the training data.
#'
#' @param train      A list containing the training set. The list must have a component `X` for predictors.
#' @param validation A list containing the validation set. The list must have a component `X` for predictors. If `NULL`, the validation set is not standardized. Default is `NULL`.
#' @param test       A list containing the test set. The list must have a component `X` for predictors. If `NULL`, the test set is not standardized. Default is `NULL`.
#'
#' @return
#' A list with the following components:
#' \item{train}{The standardized training set, with predictors centered and scaled.}
#' \item{validation}{The standardized validation set (if provided), standardized using the training set's mean and standard deviation.}
#' \item{test}{The standardized test set (if provided), standardized using the training set's mean and standard deviation.}
#'
#' @examples
#' # Example usage
#' train_data <- list(X = matrix(rnorm(100), ncol=10))
#' validation_data <- list(X = matrix(rnorm(50), ncol=10))
#' test_data <- list(X = matrix(rnorm(50), ncol=10))
#'
#' standardized <- stan(train = train_data, validation = validation_data, test = test_data)
#'
#' @export
stan = function(train, validation=NULL, test=NULL) {
  # Standardize the training data
  train_m     = colMeans(train$X)
  std_train_x = t(apply(train$X, 1, function(x) x - train_m))  
  train_sd    = apply(std_train_x, 2, 
                function(x) sqrt(x %*% x / length(x)))
  train_sd[train_sd == 0] = 1
  train$X = t(apply(std_train_x, 1, function(x) x / train_sd))  
  
  # Standardize validation set (if provided)
  if (!is.null(validation)) validation$X = 
    scale(validation$X, center = train_m, scale = train_sd)
  
  # Standardize test set (if provided)
  if (!is.null(test)) test$X = 
    scale(test$X, center = train_m, scale = train_sd)
  
  # Remove attributes from scaled datasets
  rm.att = function(x) {
    attributes(x) = attributes(x)[c(1,2)]
    x
  } 
  train$X = rm.att(train$X)
  validation$X = rm.att(validation$X)
  test$X = rm.att(test$X)
  
  # Return standardized datasets
  list(train = train, validation = validation, test = test)
}
