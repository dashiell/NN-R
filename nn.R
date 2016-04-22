# Dashiell Gough, 2016

library(R6)

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

NN = R6Class("NN",
  public = list(
    inputLayerN = 2,
    hiddenLayerN = 3,
    outputLayerN = 1,
    
    #layer weights
    W1 = NA, W2 = NA,
    yHat = NA,
    #activation function
    a2 = NA,
    z2 = NA, z3 = NA,
    lambda = 0,

    initialize = function(lambda) {
      self$lambda = lambda
            
      w1 = rnorm(self$inputLayerN*self$hiddenLayerN)
      self$W1 = matrix(data=w1,nrow=self$inputLayerN,ncol=self$hiddenLayerN)

      w2 = rnorm(self$hiddenLayerN*self$outputLayerN)
      self$W2 = matrix(data=w2, nrow=self$hiddenLayerN,ncol=self$outputLayerN)
    },
    
    #sigmoid function (element wise)
    sigmoid = function(z) {
      return (1 / (1+exp(-z)))
    },
    
    #derivative of sigmoid function -- gradient
    sigmoidPrime = function(z) {
      return ( exp(-z) /( (1+exp(-z))^2) )
    },
    
    #forward pass
    forward = function(X) {
      
      #inner (dot) product of X and W1
      self$z2 = X%*%self$W1
      self$a2 = self$sigmoid( self$z2 )
      self$z3 = self$a2%*%self$W2
      
      yHat = self$sigmoid(self$z3)
      
      return(yHat)
    },
    
    costFunction = function(X, y) {
 
      self$yHat = self$forward(X)
      J = 0.5 * sum((y - self$yHat)^2 ) / dim(X)[1] + (self$lambda/2) * (sum(self$W1^2) + sum(self$W2^2))
      return(J)
    },
    
    #derivative with respect to weights 
    costFunctionPrime = function(X, y) {
      
      self$yHat = self$forward(X)
      delta3 = -(y - self$yHat) * self$sigmoidPrime(self$z3)
      dJdW2 = t(self$a2)%*%delta3 / dim(X)[1] + self$lambda*self$W2
      
      delta2 = delta3%*%t(self$W2) * self$sigmoidPrime(self$z2)
      dJdW1 = t(X)%*%delta2 / dim(X)[1] + self$lambda*self$W1
      
      dJdW = list(dJdW1, dJdW2)
      return(dJdW)
    },
    
    #helper functions for other classes
    getParams = function() {
      return(c(self$W1, self$W2))
    },
    
    setParams = function(params) {
      W1_start = 1
      W1_end = self$hiddenLayerN * self$inputLayerN
      self$W1 = matrix(data=params[W1_start:W1_end],nrow=self$inputLayerN,ncol=self$hiddenLayerN)
      
      W2_end = W1_end + self$hiddenLayerN * self$outputLayerN
      self$W2 = matrix(data=params[W1_end+1:W2_end],nrow=self$hiddenLayerN,ncol=self$outputLayerN)
    }, 
    
    computeGradients = function(X, y) {
      list[dJdW1, dJdW2] = self$costFunctionPrime(X,y)
      return( c(dJdW1, dJdW2) )
    }
  )
)

Trainer = R6Class("Trainer",
  public = list(
    NN = NA,
    opt = NA,
    initialize = function(NN) {
      self$NN = NN
    },
    
    costWrapper = function(params, X, y) {
      self$NN$setParams(params)
      
      cost = self$NN$costFunction(X, y)
      
      return(cost)
    },
    
    gradWrapper = function(params, X, y) {
      self$NN$setParams(params)
      grad = self$NN$computeGradients(X, y)
    },
    
    train = function(X, y) {
      params = self$NN$getParams()

      opt = optim(par=params, fn=self$costWrapper, gr=self$gradWrapper, method="BFGS", X=X, y=y)
      print(opt)
      
    }
  )
)

computeNumericalGradient = function(NN, X, y) {
  paramsInitial = NN$getParams()

  numgrad = rep(0, length(paramsInitial))
  perturb = rep(0, length(paramsInitial))
  
  epsilon = 1e-4
 
  for (p in 1:length(paramsInitial)) {
    perturb[p] = epsilon
    
    NN$setParams(paramsInitial + perturb)
    loss2 = NN$costFunction(X, y)
    NN$setParams(paramsInitial - perturb)
    loss1 = NN$costFunction(X, y)
  
    #numerical gradient
    numgrad[p] = (loss2 - loss1) / (2*epsilon)
    perturb[p] = 0
  }
  NN$setParams(paramsInitial)
  
  return(numgrad)
}

trainX = matrix(c(2,4, 4,2, 9,3, 8,2.5), nrow=4,ncol=2,byrow=TRUE)
trainY = c(60,80,90,70)
#normalize
trainX = apply(trainX, 2, function(x) x = x/max(x))
trainY = trainY/100

testX = matrix(c(4,5, 5,1, 10,2, 6,3), nrow=4,ncol=2,byrow=TRUE)
testY = c(70,90,84,70)

testX = apply(testX, 2, function(x) x = x/max(x))
testY = testY/100

nn = NN$new(lambda=0.0001)

numgrad = computeNumericalGradient(nn, X, y)
grad = nn$computeGradients(X,y)

trainer = Trainer$new(nn)
trainer$train(trainX, trainY)
trainer$NN$costFunctionPrime(trainX, trainY)

yHat = nn$forward(testX)