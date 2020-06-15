%% clear all
close all;
clear all;
%% simulation setting
% obj function type
global obj_type;
obj_type = 2; % 2=b, 3=c
% parameters
m_size = 18;
n_size = 15;
global epsilon;
epsilon = 1e-3;
rng(1); % seed
global x_sym differential;
if(obj_type==2)
    global A b;
    A = rand(m_size,n_size);
    b = rand(m_size,1);
    x_sym = sym('x_sym',[n_size 1]);
    differential = gradient((b-A*x_sym).'*(b-A*x_sym),x_sym);
elseif(obj_type==3)
    global A b lamb;
    A = rand(m_size,n_size);
    b = rand(m_size,1);
    lamb = 10;
    x_sym = sym('x_sym',[n_size 1]);
    differential = gradient((b-A*x_sym).'*(b-A*x_sym)+lamb*x_sym.'*x_sym,x_sym);
end
%initial guessの生成
x_init = ones(n_size,1); % tekito
global f_list;
f_list = [f(x_init)];
%% true answer by cvx
cvx_begin
variable x_cvx(n_size)
minimize(f(x_cvx))
cvx_end
global f_cvx;
f_cvx = f(x_cvx);
%% steepest descending method
tic;
[x_fin, f_fin] = sdmo(x_init);
elapsed_time = toc;
%% plot
k_idx = 1:length(f_list(1,:));
for i=1:length(f_list(1,:))
    f_list(i) = log10(f_list(i) - f_cvx);
end
plot(k_idx,f_list); hold on;
xlabel('k', 'FontSize',18);
ylabel('$$ \log_{10}(f(x_k)-f(x^{\ast})) $$','Interpreter','latex','FontSize',18);
%% function define
% stmo main routine
function [x_fin, f_fin] = sdmo(x_init)
    global f_list;
    global f_cvx;
    global epsilon;
    itr_max = 1e5;
    alpha = 0.001;
    gamma = 0.1;
    delta = 0.1;
    x_val = x_init;
    for k=1:itr_max
        fprintf('%d,%f,%f\n',k,f(x_val)-f_cvx,x_val(1));
        x_k = x_val;
        d_k = -calcJacobi(x_k);
        if  (f(x_val) - f_cvx) <= epsilon
            fprintf('%d times iteration until convergence\n', k);
            break;
        end
        x_val = x_val + alpha*d_k;
        %alpha_armijo = armijo(alpha,x_val,d_k,@f,@calcJacobi,gamma,delta);
        %x_val = x_val + alpha_armijo*d_k;
        f_list = [f_list,f(x_val)];
    end
    x_fin = x_val;
    f_fin = f(x_val);
end

function y = f(x)
    global obj_type;
    if(obj_type==2)
        global A b;
        y = (b-A*x).'*(b-A*x);
    elseif(obj_type==3)
        global A b lamb;
        y = (b-A*x).'*(b-A*x)+lamb*x.'*x;
    end
end

function J = calcJacobi(x)
    global differential;
    global x_sym;
    J = double(subs(differential,x_sym,x));
end