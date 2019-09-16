

#' @title Super Learner with Cross-Validated Risk Prediction
#'
#' @details Please see van der Laan et al. 2007 (PMID: 17910531) & Gruber et al. 2015 (PMC4262745) for more detail
#' @param learnlib list; A list of \code{Learners} from the \code{mlr} package
#' @param Task \class{Task}; A \class{Task} from the \code{mlr} package
##' @param valset.matrix list; A list of the row numbers that correspond to the validation observations for each of the V-fold processes. Note, each list item should be a vector of rownumbers
#
SL_crossval_risk_pred <- function(learnerlib, task, valset.list){
  #.......................................................................
  # assertions
  #.......................................................................
  assert_list(valset.list)
  assert_custom_class(task, "Task")
  assert_list(learnerlib)
  assert_custom_class(learnerlib[[1]], "Learner")

  if( length(unique(unlist(valset.list))) == nrow(mlr::getTaskData(task))){
    warning("You may not have partitioned your validation sets properly. Remember, validation sets should be split into v-folds of the entire data. Would expect your validation sets to have every row of the data included.")
  }

  #.......................................................................
  # Setup pieces
  #.......................................................................
  fulldat <- mlr::getTaskData(task)
  n <- mlr::getTaskSize(task)

  #.......................................................................
  # Full Data, Z prime
  #.......................................................................
  # train base libraries
  Zprime.learnerlib.trained <- purrr::map(learnerlib,
                                          function(x, task){
                                            ret <- mlr::train(learner = x, task = task)}, task = task)

  #..............................
  # prediction matrix, Zprime
  #..............................
  Zprime <- matrix(NA, nrow = nrow(fulldat), ncol = length(learnerlib))
  for(j in 1:length(Zprime.learnerlib.trained)){
    Zprime[, j] <- get_preds(trained = Zprime.learnerlib.trained[[j]],
                        task = task)
  }



  #.......................................................................
  # Cross Validated Setup - Z
  #.......................................................................
  Z <- matrix(NA, nrow = nrow(fulldat), ncol = length(learnerlib))

  for(j in 1:ncol(Z)){ # for each algorithm (or base learner) in 1, ..., L
    for(v in 1:length(valset.list)){ # for each v in 1, ..., V
      valset <- valset.list[[v]]
      trainset <- c(1:nrow(fulldat))[! 1:nrow(fulldat) %in% valset]
      # train model
      trained <- mlr::train(learner = learnerlib[[j]],
                            task = task, subset = trainset)

      # get valset predictions
      val <- get_preds(trained = trained, task = task, subset = valset)

      # put predictions in Z
      Z[valset, j] <- val
    }
  }

  #.......................................................................
  # Minimize cross validated risk
  #.......................................................................
  algnames <- unlist( purrr::map(learnerlib, "name") )
  Y <- fulldat[, mlr::getTaskTargetNames(task)] # or really A
  if(is.factor(Y)){
    pos.class <- mlr::getTaskDesc(task)$positive
    Y <- ifelse(Y == pos.class, 1, 0)
  }

  if(length(learnerlib) == 1){ # can't do nnls on one alg
    cvrisk <- list(cvrisk = 1, coef = 1)
  } else{
    cvrisk <- nnls_cvrisk(Z = Z, Y = Y, algnames = algnames, verbose = T)
  }


  #.......................................................................
  # Aplly CVrisk Coeffs
  #.......................................................................
  if(all(cvrisk$coef == 0)){stop("The CV Risk could not be calculated for this model")}
  EL.cvrisk.preds <- Zprime %*% cvrisk$coef



  #..............................
  # outs
  #..............................
  ret <- list(cvrisk.coef = cvrisk$coef,
              alg.cvrisk.validationset = cvrisk$cvrisk,
              EL.predictions = unlist(EL.cvrisk.preds),
              task = task,
              Z = Z,
              Zprime = Zprime)

  return(ret)


}




