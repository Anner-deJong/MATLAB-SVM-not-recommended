function [Loss_Train, W] = SVM(X, y, iter, lambda, lrn_rate, epsilon)

%   Anner de Jong 05/07/2016
%
%   SVM_Train trains an SVM with given regularization and learning rate
%   requiring a seperate Loss and Gradient calculator called SVM_LG

%% Initialise hyperparameters

if ~exist('iter', 'var') || isempty(iter)
    iter = 300;
end

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0.1;
end

if ~exist('lrn_rate', 'var') || isempty(lrn_rate)
    lrn_rate = exp(-7);
end

if ~exist('epsilon', 'var') || isempty(epsilon)
    epsilon = 500;
end

%% Initialisation parameters

no_samples = size(X,1);                     % number of samples
X = [ones(no_samples,1),X];                 % include bias

X_space  = size(X,2);                       % number of input parameters (including bias)
y_space  = size(y,2);                       % number of classes

Loss_Train = zeros(1,iter);                 % initialize empty loss history
W          = zeros(X_space, y_space, iter); % initialize empty weight history

rng('default');                             % control for randomness
W(:,:,1)   = rand(X_space,y_space);         % initialize a random weight matrix


%% Iteration

for i = 1:round(iter)
    
    [loss, Grad] = SVM_LG(X,y,W(:,:,i),lambda,epsilon);     % calculate loss and gradient
    W(:,:,i+1) = W(:,:,i) - (lrn_rate * Grad);              % update weight matrix
    l_ng = loss - 0.5 * lambda * (sum(W(:,:,i).*W(:,:,i))); % calculate loss withouth regularization
    Loss_Train(i) = l_ng;                                   % update loss history

end

