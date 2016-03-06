function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly



to_reg = ones(size(theta))(:,1);
to_reg(1) = 0;



J = ( sum(((X * theta)-y).^2) + ((to_reg .* theta)' * theta).*lambda ) ./ (2*m);


grad = (X'*((X * theta)-y))/m + to_reg .* (theta .* lambda ./ m);



% ========================================================

grad = grad(:);

end
