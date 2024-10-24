#' ART_I_AM: ART-Integrated-Aggregrating Machines
#'
#' `ART_I_AM` performs adaptive and robust transfer learning through the aggregration of multiple machine learning models, 
#' specifically random forests, AdaBoost, and neural networks. This method aggregates the predictions from these models 
#' across multiple auxiliary datasets and the primary dataset to enhance model performance on the primary task. 
#' Users do not need to specify the models in the function, while the framework is general and users can write 
#' their own function integrating other machine learning models.
#'
#' @param X       A matrix for the primary dataset (target domain) predictors.
#' @param y       A vector for the primary dataset (target domain) responses.
#' @param X_aux   A list of matrices for the auxiliary datasets (source domains) predictors.
#' @param y_aux   A list of vectors for the auxiliary datasets (source domains) responses.
#' @param X_test  A matrix for the test dataset predictors.
#' @param lam     A regularization parameter for weighting the auxiliary models. Default is 1.
#' @param maxit   The maximum number of iterations for the aggregation process. Default is 5000.
#' @param eps     A convergence threshold for stopping the iterations. Default is 1e-6.
#' @param ...     Not used in ART_I_AM.
#'
#' @details
#' The `ART_I_AM` function automatically integrates three machine learning models:
#' - Random Forest (`fit_rf`)
#' - AdaBoost (`fit_gbm`)
#' - Neural Network (`fit_nnet`)
#'
#' These models are applied to both the primary dataset and auxiliary datasets. The function aggregates the predictions 
#' of each model using adaptive weights determined by the exponential weighting scheme. The aggregation improves the prediction
#' power by considering different models and data simultaneously.
#'
#' @return
#' A list containing:
#' \item{pred_ART}{The predictions for the test dataset aggregated from the different models and datasets.}
#' \item{W_ART}{The final weights for each model and dataset combination.}
#' \item{iter_ART}{The number of iterations performed until convergence.}
#'
#' @examples
# ' # Generate synthetic datasets for transfer learning
# ' dat <- generate_data(n0=50, K=1, nk=30, p=5, 
# '                      mu_trgt=1, xi_aux=0.5, ro=0.3, err_sig=1, is_test=TRUE, task="classification")
# ' # Fit ART_I_AM aggregating three models (random forest, AdaBoost, and neural network)
# ' fit <- ART_I_AM(X=dat$X, y=dat$y, X_aux=dat$X_aux, y_aux=dat$y_aux, X_test=dat$X_test)
# ' 
# ' # View the predictions and weights
# ' fit$pred_ART
# ' fit$W_ART
#' 
#' @export
ART_I_AM <- function(X, y, X_aux, y_aux, X_test, lam = 1, maxit = 5000L, 
  eps = 1e-6, ...) {

  # model setting
  if (!is.list(X_aux) || !is.list(y_aux)) {
    stop("Auxiliary sets, X_aux and y_aux, must be lists.")
  }
  K <- length(y_aux)
  if (K != length(X_aux)) {
    stop("X_aux and y_aux don't have the same length.")
  }
  n0 <- NROW(X)
  W_all <- rep(0, 3 * (K + 1))
  iter  <- 1

  # aggregation by mixing
  for (iter in seq(maxit)) {
    id0 <- sample(n0, n0 / 2)
    X_00 <- X[id0, ]
    y_00 <- y[id0]
    X_01 <- X[-id0, ]
    y_01 <- y[-id0]
    fit_0_agg1 <- fit_rf(X_00, y_00, 
      X_val = X_01, y_val = y_01, X_test=NULL)
    fit_0_agg2 <- fit_gbm(X_00, y_00, 
      X_val = X_01, y_val = y_01, X_test=NULL, dist="adaboost", n.minobsinnode=5)
    fit_0_agg3 <- fit_nnet(X_00, y_00, 
      X_val = X_01, y_val = y_01, X_test=NULL, trace=FALSE)
    W_0 <- rep(NA, 3)
    W_0[1] <- exp(-lam * fit_0_agg1$dev)
    W_0[2] <- exp(-lam * fit_0_agg2$dev)
    W_0[3] <- exp(-lam * fit_0_agg3$dev)
    W_k <- rep(0, K * 3)

    for (k in seq(K)) {
      X_k0_agg <- rbind(X_aux[[k]], X_00)
      y_k0_agg <- c(y_aux[[k]], y_00)
      fit_k_agg1 <- fit_rf(X_k0_agg, y_k0_agg, 
        X_val = X_01, y_val = y_01, X_test=NULL)
      fit_k_agg2 <- fit_gbm(X_k0_agg, y_k0_agg, 
        X_val = X_01, y_val = y_01, X_test=NULL, dist="adaboost", n.minobsinnode=5)
      fit_k_agg3 <- fit_nnet(X_k0_agg, y_k0_agg, 
        X_val = X_01, y_val = y_01, X_test=NULL, trace=FALSE)
      W_k[k*3-2] <- exp(-lam * fit_k_agg1$dev)
      W_k[k*3-1] <- exp(-lam * fit_k_agg2$dev)
      W_k[k*3-0] <- exp(-lam * fit_k_agg3$dev)
    }   
    sum_W_k <- sum(W_0) + sum(W_k)
    W_0     <- W_0 / sum_W_k
    W_k     <- W_k / sum_W_k
    W_new   <- (W_all * (iter - 1) + c(W_0, W_k)) / iter
    dif     <- sum((W_new - W_all)^2)
    W_all   <- W_new
    if (dif < eps) break
  }
  
  # final model
  fit_01 <- fit_rf(X, y, X_val = NULL, y_val = NULL, 
    X_test = X_test)
  fit_02 <- fit_gbm(X, y, X_val = NULL, y_val = NULL, 
    X_test = X_test, dist="adaboost", n.minobsinnode=5)
  fit_03 <- fit_nnet(X, y, X_val = NULL, y_val = NULL, 
    X_test = X_test, trace=FALSE)
  pred_0 <- cbind(fit_01$pred, fit_02$pred, fit_03$pred)
  pred_k <- matrix(NA, NROW(X_test), 3 * K)
  for (k in seq(K)) {
    X_k <- rbind(X_aux[[k]], X)
    y_k <- c(y_aux[[k]], y)
    fit_k1 <- fit_rf(X_k, y_k, X_val = NULL, y_val = NULL, 
      X_test = X_test)
    fit_k2 <- fit_gbm(X_k, y_k, X_val = NULL, y_val = NULL, 
      X_test = X_test, dist="adaboost", n.minobsinnode=5)
    fit_k3 <- fit_nnet(X_k, y_k, X_val = NULL, y_val = NULL, 
      X_test = X_test, trace=FALSE)
    pred_k[, k*3-2] <- fit_k1$pred
    pred_k[, k*3-1] <- fit_k2$pred
    pred_k[, k*3-0] <- fit_k3$pred
  }
  
  pred_ART <- as.vector(pred_k %*% matrix(W_k, nrow=K * 3) + 
    pred_0 %*% matrix(W_0, nrow=3))

  list(pred_ART = pred_ART, 
    W_ART = W_all, iter_ART = iter)
}
