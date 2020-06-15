%% clear all
close all;
clear all;
%% simulation setting
% parameters
m_size = 5;
n_size = 4;
global epsilon;
epsilon = 1e-4;
rng(2); % seed
global x_sym y_sym differential_x differential_y A b;
A = rand(m_size,n_size);
b = rand(m_size,1);
x_sym = sym('x_sym',[n_size 1]);
y_sym = sym('y_sym',[1 1]);
differential_x = gradient((b-A*x_sym*y_sym).'*(b-A*x_sym*y_sym),x_sym);
differential_y = gradient((b-A*x_sym*y_sym).'*(b-A*x_sym*y_sym),y_sym);
%initial guessの生成
x_init = ones(n_size+1,1); % tekito
global gradnorm_list mingradnorm_list;
disp(calcJacobi(x_init));
gradnorm_list = [norm(calcJacobi(x_init),2)^2];
mingradnorm_list = [min(gradnorm_list)];
%% steepest descending method
[x_fin, f_fin] = sdmo(x_init);
%% plot
k_idx = 1:length(gradnorm_list(1,:));
plot(k_idx,log10(mingradnorm_list)); hold on;
xlabel('k', 'FontSize',18);
ylabel('$$ \log_{10}(min(||\Delta f(x_j)||^2) $$','Interpreter','latex','FontSize',18);
%% function define
% stmo main routine
function [x_fin, f_fin] = sdmo(x_init)
    global gradnorm_list mingradnorm_list epsilon;
    itr_max = 1e5;
    alpha = 0.1;
    gamma = 0.1;
    delta = 0.1;
    x_val = x_init;
    for k=1:itr_max
        x_k = x_val;
        d_k = -calcJacobi(x_k);
        fprintf('%d,%f\n',k,min(gradnorm_list));
        if(min(gradnorm_list) <= epsilon)
            fprintf('%d times iteration until convergence\n', k);
            break;
        end
        x_val = x_val + alpha*d_k;
        %alpha_armijo = armijo(alpha,x_val,d_k,@f,@calcJacobi,gamma,delta);
        %x_val = x_val + alpha_armijo*d_k;
        gradnorm_list = [gradnorm_list,norm(calcJacobi(x_val),2)^2];
        mingradnorm_list = [mingradnorm_list,min(gradnorm_list)];
    end
    x_fin = x_val;
    f_fin = f(x_val);
end

function y = f(x)
    global A b;
    y = (b-A*x(1:end-1)*x(end)).'*(b-A*x(1:end-1)*x(end));
end

function J = calcJacobi(x)
    global x_sym y_sym differential_x differential_y;
    temp_x = subs(differential_x,x_sym,x(1:end-1));
    J = double(subs(temp_x,y_sym,x(end)));
    temp_y = subs(differential_y,x_sym,x(1:end-1));
    J = [J;double(subs(temp_y,y_sym,x(end)))];
end