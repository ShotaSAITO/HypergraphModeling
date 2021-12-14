%%% load dataset
% X: n x d dataset, where n is a number of the data points, and d is a
%    dimension of the data point
%idx: the label
%%%
[X,y] = iris_dataset;
X = X';
X = X - sum(X,1)/size(X,1)./std(X);
X = X + max(max(-X));
idx = y(1,:)' + 2*y(2,:)' + 3*y(3,:)';

addpath('./lib')

%%parameters
m=8; % the order of the uniform hypergraph
k=max(idx); % the number of clusters
gamma=1; % the coefficient of Gaussian

%%% run the algorithm
[idx_pred,~] = biclique_gaussian(X,gamma,m,k);

%%% compute error
err = 1 - AccMeasure(idx,idx_pred)/100;
fprintf('Error is %f\n', err)