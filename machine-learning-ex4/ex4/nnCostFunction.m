function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Theta1  [25 X 401]
% Theta2  [10 X 26]

a1 = [ones(m, 1) X];   % [5000 X 401]

z2 = a1 * Theta1';     % [5000 X 401] * [401 X 25]  
a2 = sigmoid(z2);      % bug, don't sigmoid the bias parameter
a2 = [ones(m, 1) a2];  % [5000 X 26]

z3 = a2 * Theta2';   % [5000 X 26] * [26 X 10]
a3 = sigmoid(z3);                 % [5000 X 10] 

Y = eye(num_labels)(y,:);   % important   [5000 X 10] 

regularization =(sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)))*lambda/(2*m);

J = sum(sum(-Y.*log(a3)-(1-Y).*log(1-a3)))/m + regularization;





% simplified code for backpropagation algorithm
%{  
delta_3 = a3 - Y;                                              % [5000 X 10]
delta_2 = (delta_3 * Theta2(:,2:end)) .* sigmoidGradient(z2);  % [5000 X 10] * [10 X 25] .* [5000 X 25] = [5000 X 25]


delta_cap1 = delta_2' * a1;    %[5000 X 25]' * [5000 X 401]
delta_cap2 = delta_3' * a2;    %[5000 X 10]' * [5000 X 26]


Theta1_grad = ((1/m) * delta_cap1) ;
Theta2_grad = ((1/m) * delta_cap2) ;
%}



% backpropagation algorithm
for t = 1:m
    % step 1
    a1 = [1 X(t, :)];       % [1 X 401]
    
    z2 = a1 * Theta1';      % [1 X 401] * [25 X 401]' = [1 X 25]
    a2 = sigmoid(z2);       % [1 X 25]  bug, don't sigmoid the bias parameter
    a2 = [1 a2];            % [1 X 26]

    z3 = a2 * Theta2';      % [1 X 26] * [10 X 26]' = [1 X 10]
    a3 = sigmoid(z3);       % [1 X 10] 
    
    % step 2
    delta3 = a3 - Y(t, :);     % [1 X 10]   Y(t) is bug!!!
    
    % step 3 
    delta2 = (delta3 * Theta2(: , 2:end)) .* sigmoidGradient(z2); %  [1 X 10]* [10 X 25] .* [1 X 25] =  [1 X 25]

    
    % step 4
    Theta1_grad = Theta1_grad + delta2' * a1; % [1 X 25]' * [1 X 401] = [25 X 401]
    Theta2_grad = Theta2_grad + delta3' * a2; % [1 X 10]' * [1 X 26] = [10 X 26]
end 

% step 5
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
