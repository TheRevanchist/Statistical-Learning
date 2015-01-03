function [error_rate, p] = logisticRegression(train, test)

X = train(:,1:9);
y = train(:,10);

[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

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

