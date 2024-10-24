#' fit_lm: Linear Regression Wrapper for the ARTtransfer package
#'
#' This function fits a linear regression model using `lm()` and returns the 
#' coefficients, deviance on a validation set, and predictions on a test set. 
#' It is specifically designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions (not used in this function but passed for compatibility). Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions (not used in this function but passed for compatibility). Default is `1-1e-5`.
#' @param ...     Additional arguments passed to the function (currently unused).
#'
#' @return
#' A list containing:
#' \item{dev}{The mean squared error (deviance) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predictions on the test set if `X_test` is provided, otherwise `NULL`.}
#' \item{coef}{The fitted coefficients of the linear model.}
#'
#' @examples
#' # Fit a linear model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- X_train %*% rnorm(5) + rnorm(100)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- X_val %*% rnorm(5) + rnorm(50)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_lm(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_lm <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, ...) {
  # fit linear regression model
  lmfit <- lm(y ~ X)
  bt  <- as.vector(coefficients(lmfit))
  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    dev <- mean((y_val - X_val %*% bt[-1] - bt[1])^2)
  }
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- X_test %*% bt[-1] + bt[1]
  }
  list(dev = dev, pred = pred, coef=bt)
}

#' fit_logit: Logistic Regression Wrapper for the ARTtransfer package
#'
#' This function fits a logistic regression model using `glm()` and returns the 
#' coefficients, deviance on a validation set, and predictions on a test set. 
#' It is specifically designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of binary responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of binary responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions. Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions. Default is `1-1e-5`.
#' @param ...     Additional arguments passed to the function (currently unused).
#'
#' @return
#' A list containing:
#' \item{dev}{The deviance (negative log-likelihood) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predicted probabilities on the test set if `X_test` is provided, otherwise `NULL`.}
#' \item{coef}{The fitted coefficients of the logistic model.}
#'
#' @examples
#' # Fit a logistic regression model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- rbinom(100, 1, 0.5)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- rbinom(50, 1, 0.5)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_logit(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_logit <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, ...) {
  # fit logistic regression model
  glmfit <- suppressWarnings(glm(y ~ X, family="binomial"))
  bt  <- coefficients(glmfit)
  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    prob <- 1 / (1 + exp(-X_val %*% bt[-1] - bt[1]))
    prob <- pmin(pmax(prob, min_prod), max_prod)
    dev <- mean(-2 * ((y_val == 0) * log(1 - prob) + 
      (y_val == 1) * log(prob)))  
  }
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- 1 / (1 + exp(-X_test %*% bt[-1] - bt[1]))
    pred <- pmin(pmax(pred, min_prod), max_prod)
  }
  list(dev = dev, pred = pred, coef = bt)
}
#' fit_glmnet_lm: Sparse Linear Regression Wrapper for the ARTtransfer package
#'
#' This function fits a sparse linear regression model using `glmnet()` from the R package glmnet for regression. 
#' It returns the coefficients, deviance on a validation set, and predictions on a test set. 
#' It is designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions (not used in this function but passed for compatibility). Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions (not used in this function but passed for compatibility). Default is `1-1e-5`.
#' @param nfolds  An integer specifying the number of folds for cross-validation. Default is 5.
#' @param ...     Additional arguments passed to the function.
#'
#' @return
#' A list containing:
#' \item{dev}{The mean squared error (deviance) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predictions on the test set if `X_test` is provided, otherwise `NULL`.}
#' \item{coef}{The fitted coefficients of the sparse linear model.}
#'
#' @examples
#' # Fit a sparse linear model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- X_train %*% rnorm(5) + rnorm(100)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- X_val %*% rnorm(5) + rnorm(50)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_glmnet_lm(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_glmnet_lm <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, nfolds = 5, ...) {
  # fit sparse regression models
  fit_cv <- cv.glmnet(y=as.vector(y), x=as.matrix(X), nfolds=nfolds)
  fit    <- glmnet(y=as.vector(y), x=as.matrix(X))
  bt <- as.vector(coefficients(fit, s=fit_cv$lambda.1se))
  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    dev  <- mean((y_val - X_val %*% bt[-1] - bt[1])^2)
  }
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- X_test %*% bt[-1] + bt[1]
  }
  list(dev = dev, pred = pred, coef = bt)
}
#' fit_glmnet_logit: Sparse Logistic Regression Wrapper for the ARTtransfer package
#'
#' This function fits a sparse logistic regression model using `glmnet()` from the R package glmnet for classification. 
#' It returns the coefficients, deviance on a validation set, and predictions on a test set. 
#' It is designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of binary responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of binary responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions. Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions. Default is `1-1e-5`.
#' @param nfolds  An integer specifying the number of folds for cross-validation. Default is 5.
#' @param ...     Additional arguments passed to the function.
#'
#' @return
#' A list containing:
#' \item{dev}{The deviance (negative log-likelihood) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predicted probabilities on the test set if `X_test` is provided, otherwise `NULL`.}
#' \item{coef}{The fitted coefficients of the sparse logistic model.}
#'
#' @examples
#' # Fit a sparse logistic regression model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- rbinom(100, 1, 0.5)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- rbinom(50, 1, 0.5)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_glmnet_logit(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_glmnet_logit <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, nfolds = 5, ...) {
  # fit sparse logistic regression models
  fit_cv <- cv.glmnet(y=as.vector(y), x=as.matrix(X), 
    family="binomial", nfolds=nfolds)
  fit    <- glmnet(y=as.vector(y), x=as.matrix(X), family="binomial")
  bt <- as.vector(coefficients(fit, s=fit_cv$lambda.1se))
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    prob <- 1 / (1 + exp(-X_val %*% bt[-1] - bt[1]))
    prob <- pmin(pmax(prob, min_prod), max_prod)
    dev <- mean(-2 * ((y_val == 0) * log(1 - prob) + 
      (y_val == 1) * log(prob)))  
  }
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- 1 / (1 + exp(-X_test %*% bt[-1] - bt[1]))
    pred <- pmin(pmax(pred, min_prod), max_prod)
  }
  list(dev = dev, pred = pred, coef = bt)
}
#' fit_rf: Random Forest Wrapper for the ARTtransfer package
#'
#' This function fits a random forest classification model using `randomForest()` from the R package randomForest. 
#' It returns the deviance on a validation set 
#' and predictions on a test set. It is designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of binary responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of binary responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions. Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions. Default is `1-1e-5`.
#' @param ...     Additional arguments passed to the `randomForest()` function.
#'
#' @return
#' A list containing:
#' \item{dev}{The deviance (negative log-likelihood) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predicted probabilities on the test set if `X_test` is provided, otherwise `NULL`.}
#'
#' @examples
#' # Fit a random forest model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- rbinom(100, 1, 0.5)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- rbinom(50, 1, 0.5)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_rf(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_rf <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, ...) {
  # fit random forest model
  fit <- randomForest(X, as.factor(y))
  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    prob <- as.vector(predict(fit, X_val, type="prob")[,2])
    prob <- pmin(pmax(prob, min_prod), max_prod)
    dev <- mean(-2 * ((y_val == 0) * log(1 - prob) + 
      (y_val == 1) * log(prob)))  
  }
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- as.vector(predict(fit, X_test, type="prob")[,2])
    pred <- pmin(pmax(pred, min_prod), max_prod)
  }
  list(dev = dev, pred = pred)
}
#' fit_gbm: Gradient Boosting Wrapper for the ARTtransfer package
#'
#' This function fits a gradient boosting model using `gbm()` from the R package gbm. 
#' It returns the deviance on a validation set 
#' and predictions on a test set. It is designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of binary responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of binary responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions. Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions. Default is `1-1e-5`.
#' @param ...     Additional arguments passed to the `gbm()` function.
#'
#' @return
#' A list containing:
#' \item{dev}{The deviance (negative log-likelihood) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predicted probabilities on the test set if `X_test` is provided, otherwise `NULL`.}
#'
#' @examples
#' # Fit a gradient boosting model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- rbinom(100, 1, 0.5)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- rbinom(50, 1, 0.5)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_gbm(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_gbm <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, ...) {
  # fit gradient boosting model
  train_dat <- as.data.frame(cbind(y=c(0,1)[factor(y)], X))
  fit <- gbm(y~., data=train_dat, n.trees=500, ...)

  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    val_dat <- as.data.frame(cbind(y=NA, X_val))
    prob <- as.vector(predict(fit, val_dat, n.trees=500, type="response"))
    prob <- pmin(pmax(prob, min_prod), max_prod)
    dev <- mean(-2 * ((y_val == 0) * log(1 - prob) + 
      (y_val == 1) * log(prob)))  
  }
  pred <- NULL
  if (!is.null(X_test)) {
    test_dat <- as.data.frame(cbind(y=NA, X_test))
    pred <- as.vector(predict(fit, test_dat, n.trees=500, type="response"))
    pred <- pmin(pmax(pred, min_prod), max_prod)
  }
  list(dev = dev, pred = pred)
}
#' fit_nnet: Neural Network Wrapper for the ARTtransfer package
#'
#' This function fits a neural network model using `nnet()` from the R package nnet. 
#' It returns the deviance on a validation set 
#' and predictions on a test set. It is designed for use in the `ART` adaptive and robust transfer learning framework.
#'
#' @param X       A matrix of predictors for the training set.
#' @param y       A vector of binary responses for the training set.
#' @param X_val   A matrix of predictors for the validation set. If `NULL`, deviance is not calculated.
#' @param y_val   A vector of binary responses for the validation set. If `NULL`, deviance is not calculated.
#' @param X_test  A matrix of predictors for the test set. If `NULL`, predictions are not generated.
#' @param min_prod A numeric value indicating the minimum probability bound for predictions. Default is `1e-5`.
#' @param max_prod A numeric value indicating the maximum probability bound for predictions. Default is `1-1e-5`.
#' @param ...     Additional arguments passed to `nnet()`.
#'
#' @return
#' A list containing:
#' \item{dev}{The deviance (negative log-likelihood) on the validation set if provided, otherwise `NULL`.}
#' \item{pred}{The predicted probabilities on the test set if `X_test` is provided, otherwise `NULL`.}
#'
#' @examples
#' # Fit a neural network model with validation and test data
#' X_train <- matrix(rnorm(100 * 5), 100, 5)
#' y_train <- rbinom(100, 1, 0.5)
#' X_val <- matrix(rnorm(50 * 5), 50, 5)
#' y_val <- rbinom(50, 1, 0.5)
#' X_test <- matrix(rnorm(20 * 5), 20, 5)
#' 
#' fit <- fit_nnet(X_train, y_train, X_val, y_val, X_test)
#'
#' @export
fit_nnet <- function(X, y, X_val, y_val, X_test, 
    min_prod = 1e-5, max_prod = 1-1e-5, ...) {
  # Fit a neural network model
  fit <- nnet(X, y, size=10, 
    rang=0.1, decay=5e-4, maxit=200, MaxNWts=1e5, ...)
  
  dev <- NULL
  if ((!is.null(X_val)) & (!is.null(y_val))) {
    prob <- as.vector(predict(fit, X_val))
    prob <- pmin(pmax(prob, min_prod), max_prod)
    dev <- mean(-2 * ((y_val == 0) * log(1 - prob) + 
      (y_val == 1) * log(prob)))
  }
  
  pred <- NULL
  if (!is.null(X_test)) {
    pred <- as.vector(predict(fit, X_test))
    pred <- pmin(pmax(pred, min_prod), max_prod)
  }
  
  list(dev = dev, pred = pred)
}


