% This script does a 5-fold cross validation using normalized logistic
% regression. It classified the observations of the Breast Cancer Wisconsin
% database. You only need to run it.

data = load('bcw.txt');
data(any(isnan(data),2),:) = [];
input = data(:, 2:11);
for i = 1:683
    if input(i,10) == 2
        input(i,10) = 0;
    else
        input(i,10) = 1;
    end
end

input = input(randperm(size(input,1)),:);

% initialize variables
error_rate = zeros(5,1);
l = length(input)/5;
input1 = input(1:l,:);
input2 = input(l+1:2*l,:);
input3 = input(2*l+1:3*l,:);
input4 = input(3*l+1:4*l,:);
input5 = input(4*l+1:5*l,:);

cross1 = [input2' input3' input4' input5']';
[error_rate(1,1), p] = normalizedLogisticRegression(cross1, input1);


cross2 = [input1' input3' input4' input5']';
[error_rate(2,1), p] = normalizedLogisticRegression(cross2, input2);

cross3 = [input1' input2' input4' input5']';
[error_rate(3,1), p] = normalizedLogisticRegression(cross3, input3);

cross4 = [input1' input2' input3' input5']';
[error_rate(4,1), p] = normalizedLogisticRegression(cross4, input4);

cross5 = [input1' input2' input3' input4']';
[error_rate(5,1), p] = normalizedLogisticRegression(cross5, input5);

err = sum(error_rate)/5
