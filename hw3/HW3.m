% Time step
delta_t = 1;

% System Matrices
A = [1, delta_t, 0, 0; 
     0, 1, 0, 0; 
     0, 0, 1, delta_t; 
     0, 0, 0, 1];
B = [0.5 * delta_t^2, 0; 
     delta_t, 0; 
     0, 0.5 * delta_t^2; 
     0, delta_t];
C = [1, 0, 0, 0; 
     0, 0, 1, 0];
Qvar = 1
Rvar = 0.1
Q = diag([Qvar, Qvar, Qvar, Qvar]);
R = diag([Rvar, Rvar]);

% Initial state and error covariance
x_est = [0; 0; 0; 0];
P_est = diag([1, 1, 1, 1]);

% Number of time steps
N = 20;

% Simulating control inputs (accelerations)
u = [[.1];[.1]];

% True state simulation (for generating measurements)
true_state = zeros(4, N);
for k = 2:N
    true_state(:, k) = A * true_state(:, k-1) + B * u;
end

% Generating noisy measurements
measurement_noise = sqrt(R) * randn(2, N);
y = C * true_state + measurement_noise;

% Kalman Filter Implementation
estimated_state = zeros(4, N);
for k = 1:N
    % Predict
    x_pred = A * x_est + B * u;
    P_pred = A * P_est * A' + Q;

    % Update
    y_residual = y(:, k) - C * x_pred;
    S = C * P_pred * C' + R;
    K = P_pred * C' / S;
    x_est = x_pred + K * y_residual;
    P_est = (eye(4) - K * C) * P_pred;

    % Display the estimated position
    fprintf('Time Step %d: True Position = [%.2f, %.2f]\n', k, true_state(1, k), true_state(3, k));
    fprintf('Time Step %d: Estimated Position = [%.2f, %.2f]\n', k, x_est(1), x_est(3));
    
    % Optimal predictor
    estimated_state(:, k) = x_est;
end
predicted_state = A * estimated_state + B * u;

% Plotting
figure;
plot(true_state(1, :), true_state(3, :), 'g', 'DisplayName', 'True Path', 'LineWidth', 1.5);
hold on;
plot(y(1, :), y(2, :), 'rx', 'DisplayName', 'Measurements', 'MarkerSize', 8);
plot(estimated_state(1, :), estimated_state(3, :), 'b', 'DisplayName', 'Estimated Path', 'LineWidth', 1.5);
plot(predicted_state(1, :), predicted_state(3, :), 'k--', 'DisplayName', 'Predicted Path', 'LineWidth', 1.5);
xlabel('Position X', 'FontSize', 14);
ylabel('Position Y', 'FontSize', 14);
% title('Kalman Filter and Predictor Performance', 'FontSize', 16);
legend('Location', 'NorthWest', 'FontSize', 12);
grid on;
saveas(gcf, 'q1-r0.1.png');

