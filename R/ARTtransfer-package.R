#' ARTtransfer: Adaptive and Robust Transfer Learning for Enhanced Model Performance
#'
#' The ARTtransfer package implements the Adaptive and Robust Transfer Learning (ART) framework 
#' introduced by Wang et al. (2023, DOI: 10.1002/sta4.582). This framework enhances model performance 
#' on a primary task (target domain) by utilizing information from auxiliary datasets (source domains). 
#' The package is specifically designed to prevent negative transfer, ensuring that auxiliary data 
#' does not degrade model performance.
#' 
#' ARTtransfer is a flexible and general framework that also includes variable importance metrics, 
#' enabling users to evaluate the contribution of each variable to the final model and improve the 
#' interpretability of results.
#' 
#' The package includes implementations of common predictive models such as linear regression, 
#' logistic regression, lasso and elastic net penalized regression (both linear and logistic), 
#' random forest, gradient boosting machines, and neural networks. Users can also define and integrate 
#' their own predictive models into the ART framework, based on the provided examples.
#' 
#' In addition, the package implements a framework ART-I-AM that aggregates multiple machine learning 
#' methods into the ART process. This is exemplified with implementations of random forests, AdaBoost, 
#' and a basic neural network. Users can further expand the framework by incorporating 
#' other predictive models of their choice.
#' 
#' @section Functions:
#' - \code{ART()}: Main function for performing adaptive and robust transfer learning.
#' - \code{generate_data()}: Generates synthetic datasets for transfer learning simulations.
#' - Wrapper functions: Functions like \code{fit_lm()}, \code{fit_logit()}, and \code{fit_random_forest()} 
#'   are used to fit models in the ART framework.
#'
#' @section Examples:
#' To perform ART on synthetic data:
#' \code{
#' dat <- generate_data(n0=100, K=3, nk=50, is_noise=TRUE, p=10)
#' fit <- ART(dat$X, dat$y, dat$X_aux, dat$y_aux, dat$X_test, func=fit_lm, lam=1)
#' }
#'
#' For more details, see the documentation for individual functions.
#' @importFrom stats coefficients glm lm predict rnorm rbinom
#' @importFrom glmnet glmnet cv.glmnet
#' @importFrom gbm gbm
#' @importFrom nnet nnet
#' @importFrom randomForest randomForest
#' 
#' @name ARTtransfer
#' @keywords internal
"_PACKAGE"

