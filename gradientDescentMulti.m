function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  htheta = X*theta;
  theta0 = theta(1) - alpha/m *sum((htheta - y).*X(:,1));
  theta1 = theta(2) - alpha/m *sum((htheta - y).*X(:,2));
  theta2 = theta(3) - alpha/m *sum((htheta - y).*X(:,3));
  theta = [theta0;theta1;theta2] 
  J_history(iter) = computeCostMulti(X, y, theta);

end

end
