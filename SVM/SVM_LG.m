function [Loss, Grad] = SVM_LG(X, y, W, lambda, epsilon)

%   Anner de Jong 05/07/2016
%
%   SVM_LG calculates the Loss and the Gradients for an SVM machine
%   learning algorithm, using a one vs all approach

%   Classification:
%   Loss function for one sample of output size y: 1 x g
%   Loss = sum_over_g(max(1, y_g - y_correct)

%   Regression:
%   if the difference is outside of regression band (epsilon):
%   Squared of (difference between real output and calculated output minus
%   epsilon)
%   Loss = if (abs(y_train - y_real) > epsilon):
%             (abs(y_train - y_real) - epsilon) ^2
%   else 0

%% Initialisation

global type
Loss = 0;
Grad = zeros(size(W));

no_samples = size(X,1);                                 % number of training examples
sco = X*W;                                              % calculate the output matrix

%% Regression

if strcmp(type,'regression')
    % loss_matrix (actually array) is the calculated X*W matrix of scores
    
    % precalculation
    loss_matrix = max(abs(sco-y),epsilon);
    loss_matrix = loss_matrix - epsilon;

    % right now all entries are positive due to abs, correct for negative grad:
    corr = zeros(size(loss_matrix));
    corr((sco-y)>0) = 2;
    corr = corr - 1;
    loss_matrix = corr .* loss_matrix;
    
    % calculate loss
    Loss = sum(0.5*loss_matrix.^2) / no_samples;        % correct sum to average per sample
    Loss = Loss + 0.5 * lambda * (sum(W.*W));           % correct for regularization
    
    % calculate gradient   
    Grad             = (X'*loss_matrix) / no_samples;   % correct sum to average per sample
    Grad             = Grad + lambda * W;               % correct for regularization
    
    return
end

%% Classification

%% Precalculation
% loss_matrix is the calculated X*W matrix of scores, converted into the
% losses according to SVM

[~,corr_sco_ind] = max(y,[],2);                         % identify the correct score indices

idL = sub2ind(size(sco), 1:no_samples, corr_sco_ind');  % retrieve the calculated value
train_score = sco(idL)';                                % for the index of the correct score

train_score      = train_score - 1;                     % SVM: subtract 1 before next step
loss_matrix      = bsxfun(@minus,sco,train_score);      % subtract calculated value for true index from other indices
loss_matrix(idL) = 0;                                   % correct for the correct score indices themselves
loss_matrix      = max(loss_matrix,0);

%% Calculate the loss

Loss = sum(sum(loss_matrix)) / no_samples;              % correct sum to average per sample
Loss = Loss + 0.5 * lambda * sum(sum(W.*W));            % correct for regularization

%% Calculate the gradient

%loss_yn       = double(bsxfun(@gt,correct_losses,0));   % convert all positive losses into 1's
loss_matrix(loss_matrix>0) = 1;                         % convert all positive losses into 1's
corr_sco_gr      = sum(loss_matrix,2);
loss_matrix(idL) = -corr_sco_gr;

Grad             = (X'*loss_matrix) / no_samples;       % correct sum to average per sample
Grad             = Grad + lambda * W;                   % correct for regularization

end

