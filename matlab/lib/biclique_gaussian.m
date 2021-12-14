function [idx,W] = biclique_gaussian(X,gamma,m,k)
%%%%%%%%%%%%%%%%%%%
% Copyright: Shota Saito, author of the following paper: 
%           Hypergraph Embedding via Spectral Connection: Cut, Weighted
%           Kernel k-means, and Heat Kernel, AAAI-22
%
% This program is to compute Alg.1 with a biclique kernel whose base kernel
% is Gaussian kernel.
% Input:
%    X: an n x d matrix containing data, where n is a number of data, and d is a dimenison of data 
%    gamma: the parameter for gaussian kernel.
%    m: the order of uniform hypergraph we want to embed.
%    k: the number of clusters
% Output:
%    idx: an n x 1 vector cotaining predicted clusters.
%    W: n x n gram matrix for a biclique kernel
%%%%%%%%%%%%%%%%%%%

n = size(X,1);

%%%the first line of Alg.1
%Computing the gram matrix for a base kernel, which is Gaussian kernel
nsq=sum(X.^2,2);
K=bsxfun(@minus,nsq,(2*X)*X.');
K=bsxfun(@plus,nsq.',K);
K=exp(-gamma*K);

%%%the second line of Alg.1
%compuing \delta as D in Eq.(6)
d = sum(K);
D = repmat(d',1,n);

%computing \rho as E in Eq.(6)
e = sum(d);
E = e + 0*K;%by this operation we can make E as a matrix whose all the elements are e.

%computing K^{(m)} as E by using Eq.(6)
%this can be done in O(n^2) time, together with computing D and E.
%See details in the main text
W = n^2*K + (m-2)*n*D/2 + (m-2)*n*D'/2 + (m-2)^2*E/4;

%%%the third line of Alg.1
% compute square root of Dv as Dv 
d = sqrt(sum(W,1));
Dv = zeros(n);
for i = 1:n
    Dv(i,i) = 1/d(i);
end
%obtain top k eigenvectors
A = Dv*W*Dv;
[vec,~] = eig(A,'nobalance');
evecs = vec(:,1:k);

%%%the forth step of Alg.1
%conducting k-means to top k eigenvectors
idx = kmeans(evecs,k,'emptyaction','singleton');
end