function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
minC=Inf;
minError=Inf;
minSigma= Inf;
sigma = 0.3;
ar=[0.01,0.03,0.1,0.3,1,3,10,30];
ar=ar(:)
for i = 1:8,
    C=ar(i);
    for j=1:8,
        sigma=ar(j);
        model=svmTrain(X,y,C,@(x1,x2)gaussianKernel(x1,x2,sigma));
        predictions=svmPredict(model,Xval);
        curError= mean(double(predictions~=yval));

        if(curError<minError)
            minC=C;
            minSigma=sigma;
            minError=curError;
        end
    end
end

C=minC;
sigma=minSigma;

        

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
