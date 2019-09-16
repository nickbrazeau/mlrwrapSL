#' @title Get Predictions from Learner
#' @param trained \class{WrappedModel}; A \code{mlr} learner that has been trained with \code{mlr::train} and results in a \class{WrappedModel}
#' @param task \class{Task}; A \code{mlr} task
#' @param subset <numeric vector>; A numeric vector of the rows in the \class{Task} data that should be used for training
#' @details Function to extract predictions as a list from the various individually trained learners


get_preds <- function(trained, task, subset=NULL){
  if(mlr::getTaskType(task) == "classif"){

    # pull details from mlr
    pos.class <- mlr::getTaskDesc(task)$positive
    preds.i <- predict(trained, task = task, subset = subset)
    preds.i <- mlr::getPredictionProbabilities(pred = preds.i, cl = pos.class)

  } else if(mlr::getTaskType(task) == "regr"){
    preds.i <- predict(trained, task = task, subset = subset)$data$response
  }
  return(preds.i)
}


#' @title nnls_cvrisk
#' @details This is the Cross validated risk.
#' @details Note the function is essentially  lifted directly from SuperLearner::method.NNLS. I dropped observation weights and changed the return slightly
#' @param Z numeric matrix; the matrix of predictions from the K algorithms
#' @param Y numeric; outcome
#' @param algnames <character vector>; names of the K algorithms that correspond to the columns in \param{Z}
#' @param verbose logical; Talk or quiet
#'

nnls_cvrisk <- function (Z, Y, algnames, verbose = T){
  cvRisk <- apply(Z, 2, function(x){return( mean((x - Y)^2) )})

  names(cvRisk) <- algnames
  fit.nnls <- nnls::nnls( Z, Y)
  if (verbose) {
    message(paste("Non-Negative least squares convergence:",
                  fit.nnls$mode == 1))
  }
  initCoef <- coef(fit.nnls)
  initCoef[is.na(initCoef)] <- 0
  if (sum(initCoef) > 0) {
    coef <- initCoef/sum(initCoef)
  }
  else {
    warning("All algorithms have zero weight", call. = FALSE)
    coef <- initCoef
  }
  out <- list(cvrisk = cvRisk, coef = coef)
  return(out)
}
