function [error_rate, p] = normalizedLogisticRegression(train, test)

X = train(:,1:9);
y = train(:,10);

[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

options = optimset('GradObj', 'on', 'MaxIter', 400);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


testX = test(:,1:9);
[m, n] = size(testX);
testX = [ones(m, 1) testX];

testY = test(:,10);
p = predict(theta, testX);

a = 0;
for i = 1:m
    if p(i) == testY(i)
        a = a+1;
    end
end    

error_rate = 1 - (a/m);

