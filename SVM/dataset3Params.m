function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

C_options=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma_options=[0.01 0.03 0.1 0.3 1 3 10 30];



minerror=100;
C_ideal=C_options(1);
sigma_ideal=sigma_options(1);

for i=1:1:length(C_options)
    for j=1:1:length(sigma_options)
        C=C_options(i);
        sigma=sigma_options(j);
        model=svmTrain(X,y,C,@(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions=svmPredict(model,Xval);

        errorfn=mean(double(predictions~=yval));
        if(errorfn<minerror)
            minerror=errorfn;
            C_ideal=C;
            sigma_ideal=sigma;
        end
    end
end

C=C_ideal;
sigma=sigma_ideal;


% =========================================================================

end
