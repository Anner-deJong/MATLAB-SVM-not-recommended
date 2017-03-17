function [Loss_Test, ccr] = SVM_Test(X, y, W, iter, epsilon)

%   Anner de Jong 05/07/2016
%
%   Run after SVM_Train
%   SVM_Test calculates the SVM loss for every W (1 for each iter)

%% Initialise hyperparameters

if ~exist('iter', 'var') || isempty(iter)
    iter = 300;
end

if ~exist('epsilon', 'var') || isempty(epsilon)
    epsilon = 500;
end

%% Initialisation parameters

no_samples = size(X,1);         % number of samples
X = [ones(no_samples,1),X];     % include bias

Loss_Test = zeros(1,iter);      % initialize empty loss history
ccr       = zeros(1,iter);      % initialize empty correct classification rate history


%% Iteration

for i = 1:round(iter)
    
    pred          = X*W(:,:,i);                          % partial calculate for test loss
      % CHECK THIS IMPLEMENTATION OF DIFF/CCR
    diff          = max(abs(pred-y) - epsilon, 0);       % partial calculate for test loss
    ccr(i)        = sum(diff==0) / no_samples;           % correct classification rate
    Loss_Test(i)  = sum(0.5*diff.^2) / no_samples;       % Test loss   

end

sum(diff)
