import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ctr
from ipywidgets import interact, FloatSlider
from scipy.signal import place_poles
import mplcursors


## Constantes
g = 9.81 # m/s²
m = 2.4 # kg
u0 = 23.43245 # m/s
theta0 = 2.559*180/np.pi # rad, considerar theta
phi0 = 3*180/np.pi # rad

Ixx = 0.05598  # kgm²
Iyy = 0.04788  # kgm²
Izz = 0.10352  # kgm²
Ixz = -0.00211 # kgm²
Ixy = -0.0000075 # kgm²
Iyz = -0.0000002 # kgm²

## Funcoes
def dbeta_dp(Yp, Yr, Ybeta, Lp, Lr, Lbeta, Np, Nr, Nbeta): # Mudança de beta pela rolagem, provem de derivacao das eqs de lateral
    denominador = Ybeta + Yr * ((Lr * Nbeta - Lbeta * Nr) / (Lr * Np - Lp * Nr))
    numerador = Yp + Yr * ((Lp * Nbeta - Lbeta * Np) / (Lr * Np - Lp * Nr))
    resultado = -numerador / denominador
    return resultado

def dbeta_dr(Yp, Yr, Ybeta, Lp, Lr, Lbeta, Np, Nr, Nbeta): # Mudança de beta pela guinada, provem de derivacao das eqs de lateral
        numerador = (Np * Lr / Lp) - Nr
        denominador = Nbeta - (Np * Lbeta / Lp)      
        resultado = numerador / denominador
        return resultado

# Resposta de estados
def resposta_livre(A_sistema,  x0, t_span, t_eval, titulo, estados = ['$u$ (m/s)', '$\\alpha$ (rad)', '$q$ (rad/s)', '$\\theta$ (rad)', '$\\beta$ (rad)', '$p$ (rad/s)', '$r$ (rad/s)', '$\phi$ (rad)']):
    sol = solve_ivp(
        lambda t, x: A_sistema @ x,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method='RK45'
    )

    plt.figure()
    for i in estados:
        pos_estado = nomes_x.index(i)
        plt.plot(sol.t, sol.y[pos_estado, :], label=i)
    plt.title(f'Resposta Livre dos Estados - {titulo}')
    plt.xlabel('Tempo (s)')
    plt.xlim(0,10)
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    mplcursors.cursor(hover=True)

    plt.show()

    return sol

def diagrama_polos(A_sistema, titulo):
    autovalores = np.linalg.eigvals(A_sistema)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(autovalores), np.imag(autovalores), color='red', marker='o')
    plt.axhline(0, color='black', linewidth=1.0, linestyle='--')
    plt.axvline(0, color='black', linewidth=1.0, linestyle='--')
    plt.title(f'Diagrama de Polos - {titulo}')
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.grid(True)
    plt.show()
    return autovalores

# Atuadores
def esforco_atuadores(solucao, K_ganho, titulo, plot = False):
    U = -K_ganho @ solucao.y
    if plot == True:
        plt.figure(figsize=(10, 6))
        for i in range(U.shape[0]):
            plt.plot(solucao.t, U[i, :], label=nomes_u[i])
        plt.title(f'Esforço dos Atuadores - {titulo}')
        plt.xlabel('Tempo (s)')
        plt.xlim(0,10)
        plt.ylabel('Entrada de Controle')
        plt.grid(True)
        plt.legend()
        plt.show()
        pico_esforco = np.max(np.abs(U), axis=1)
        for i, pico in enumerate(pico_esforco):
            print(f'Pico {nomes_u[i]}: {pico:.2f}')
    return U

# Observador
def observacao(polos_obs, K_sis, titulo, xhat0 = 0.2*np.ones(8), estados = ['$u$ (m/s)', '$\\alpha$ (rad)', '$q$ (rad/s)', '$\\theta$ (rad)', '$\\beta$ (rad)', '$p$ (rad/s)', '$r$ (rad/s)', '$\phi$ (rad)']):
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    # Calcula a matriz do observador
    observ = place_poles(A.T, C.T, polos_obs)
    L = observ.gain_matrix.T   
    Ahat = A - B @ K_sis - L @ C

    # Dinâmica combinada do sistema e observador
    def blocos_obs(t, z):
        x = z[:8]
        xhat = z[8:]
        u = -K_sis @ x
        dx = A @ x + B @ u
        y = C @ x
        dxhat = A @ xhat + B @ u + L @ (y - C @ xhat)
        return np.concatenate((dx, dxhat))

    # Condições iniciais
    z0 = np.concatenate((x0, xhat0))
    
    # Resolve o sistema
    sol_obs = solve_ivp(blocos_obs, t_span, z0, t_eval=t_eval, method='RK45')
    
    # Calcula erro de observação
    dx = sol_obs.y[:8, :]
    dxhat = sol_obs.y[8:, :]
    erro = dx - dxhat

    # Gráfico conjunto de todos os erros 
    plt.figure(figsize=(10, 6))
    for i, estado in enumerate(estados):
        plt.plot(sol_obs.t, erro[i, :], label=estado)
    plt.title(f'Erro de Observação - {titulo}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro')
    plt.legend()
    plt.grid(True)
    mplcursors.cursor(hover=True)
    plt.tight_layout()
    plt.show()

    # Subplots individuais 
    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    axs = axs.ravel()
    for i in range(8):
        axs[i].plot(sol_obs.t, dx[i, :], label=estados[i])
        axs[i].plot(sol_obs.t, dxhat[i, :], linestyle = '--', label='Estimação ' + estados[i])
        axs[i].set_title(estados[i])
        axs[i].set_xlabel('Tempo (s)')
        axs[i].set_ylabel('Erro')
        axs[i].legend()
        axs[i].grid(True)
    plt.suptitle(f'Erro de Observação por Estado - {titulo}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.legend()
    plt.show()
    obs_eig = diagrama_polos(Ahat, f'Erro e(t) - {titulo}')
    return(erro)

# Seguidor de referência
def seguidor_LQR(A, B, K_lqr, Q, R, P, x0, x_r = np.zeros(8), estados= ['$u$ (m/s)', '$\\alpha$ (rad)', '$q$ (rad/s)', '$\\theta$ (rad)', '$\\beta$ (rad)', '$p$ (rad/s)', '$r$ (rad/s)', '$\phi$ (rad)'], entradas=["$\delta_e$ ($^o$)", "$\delta_r$ ($^o$)", "$\delta_a$ ($^o$)"]):
    t_span = (0, 11)
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Retroalimentação com eta
    def d_eta(t, eta):
        return - (A - B @ K_lqr).T @ eta - Q @ x_r

    eta_T = Q @ x_r
    sol_eta = solve_ivp(d_eta, [t_eval[-1], t_eval[0]], eta_T, t_eval=t_eval[::-1])
    eta_t = sol_eta.y[:, ::-1]  # reverter para frente no tempo

    # Interpolador pra ficar mais fácil
    def eta_interp(idx):
        return eta_t[:, idx]

    # Dinâmica do seguidor
    def dx_seguidor(t, x):
        idx = np.searchsorted(t_eval, t)
        eta = eta_interp(idx)
        u = -K_lqr @ (x-x_r) + np.linalg.inv(R) @ B.T @ (eta - P @ x_r)
        return A @ x + B @ u

    sol = solve_ivp(dx_seguidor, t_span, x0, t_eval=t_eval, method='RK45')

    # Plotagem
    plt.figure()
    for i, estado in enumerate(estados[:sol.y.shape[0]]):
        plt.plot(sol.t, sol.y[i, :], label=estado)
    plt.title('Resposta dos Estados - Seguidor LQR com Pré-Alimentação')
    plt.xlabel('Tempo (s)')
    plt.xlim(t_span[0],t_span[1]-1)
    plt.ylabel('Amplitude')
    plt.ylim(-0.3,0.3)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    U = np.zeros((B.shape[1], len(t_eval)))
    for i, t in enumerate(t_eval):
        eta = eta_t[:, i]
        x = sol.y[:, i]
        u = -K_lqr @ x + np.linalg.inv(R) @ B.T @ (eta - P @ x_r)
        U[:, i] = u

    plt.figure()
    for i, entrada in enumerate(entradas[:U.shape[0]]):
        plt.plot(t_eval, U[i, :], label=entrada)
    plt.title('Esforço dos Atuadores - Seguidor LQR com Pré-Alimentação')
    plt.xlabel('Tempo (s)')
    plt.xlim(t_span[0],t_span[1]-1)
    plt.ylabel('Entrada de Controle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sol, U

## Determinados do XFLR5
Xu=	-0.17604
Xalpha=	1.3633
Zu=	-1.9892
Zalpha=	-18.566
Zq=	-4.9505
Mu=	0.0010976
Malpha=	-1.1415
Mq=	-0.82869

	
Ybeta=	-0.32304
Yp=	-0.068839
Yr=	0.15623
Lbeta=	-0.3551
Lp=	-1.1237
Lr=	0.18423
Nbeta=	0.13096
Np=	-0.099993
Nr=	-0.045389


## Calculados por proxy
    
db_dp = dbeta_dp(Yp, Yr, Ybeta, Lp, Lr, Lbeta, Np, Nr, Nbeta)
db_dr = dbeta_dr(Yp, Yr, Ybeta, Lp, Lr, Lbeta, Np, Nr, Nbeta)
Xq = 0
Xbeta =  (2.220175-2.2207)/0.25 # dFx/dbeta, do XFLR5
Xp = Xbeta * db_dp # dCx/dbeta * dbeta_dp
Xr = Xbeta * db_dr

Zbeta = (23.30638-23.32382)/0.25
Zp = Zbeta*db_dp
Zr = Zbeta*db_dr

Mbeta = (-0.01431499 +0.000351469)/0.25
Mp = Mbeta * db_dp
Mr = Mbeta * db_dr

## Para o controle, estimados individualmente no XFLR5

# Subscrito δelevador
X_de=	-2.4249
Y_de=	0.0040723
Z_de=	-52.927
L_de=	-0.00034691
M_de=	-14.969
N_de=	-0.00087569


# Subscrito δrudder (leme)
X_dr =	 -96.58
Y_dr = 	-0.0022158
Z_dr =	 -384.27
L_dr = 	0.0012264
M_dr =	 -11.459
N_dr =	 -0.01483


# Subscrito δaileron
X_da=	-2.7749
Y_da=	0.00056597
Z_da=	-128.04
L_da=	-0.00005845
M_da=	-22.489
N_da=	0.000060727

# Matrizes intermediarias

F = np.array([
    [Xu / m, Xalpha / m, Xq / m, -g * np.cos(theta0) * np.cos(phi0), Xbeta / m, Xp / m, Xr / m, 0],
    [Zu / m, Zalpha / m, Zq/m + u0, g * np.sin(theta0) * np.cos(phi0),
     Zbeta / (m), Zp / (m), Zr / (m), g * np.sin(phi0)],
    [Mu, Malpha, Mq, 0, Mbeta, Mp, Mr, 0],
    [0, 0, np.cos(phi0), 0, 0, 0, -np.sin(phi0), 0],
    [0, 0, 0, 0, Ybeta / (m ), Yp / (m ), (Yr) / (m) - u0, g * np.cos(phi0) * np.cos(theta0)],
    [0, 0, 0, 0, Lbeta, Lp, Lr, 0],
    [0, 0, 0, 0, Nbeta, Np, Nr, 0],
    [0, 0, np.tan(theta0) * np.sin(phi0), 0, 0, 1, np.tan(theta0) * np.cos(phi0), 0]
])

E = np.array([
    [1, 0, 0, 0,  0,  0, 0, 0],
    [0, 1, 0, 0,  0,  0, 0, 0],
    [0, 0, Iyy, 0,  -Ixy,  -Iyz, 0, 0],
    [0, 0, 0, 1,  0,  0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, -Ixy, 0,  0, Ixx,  -Ixz, 0],
    [0, 0, Iyz, 0, 0, -Ixz,  Izz, 0],
    [0, 0, 0, 0,  0,   0,  0, 1]
])


G = np.array([
    [X_de/m, X_dr/m, X_da/m],
    [Z_de/(m*u0), Z_dr/(m*u0), Z_da/(m*u0)],
    [M_de, M_dr, M_da],
    [0, 0, 0],
    [Y_de/(m*u0), Y_dr/(m*u0), Y_da/(m*u0)],
    [L_de, L_dr, L_da],
    [N_de, N_dr, N_da],
    [0, 0, 0]
])

# Para o Espaço de Estados

A = np.linalg.inv(E) @ F
B = np.linalg.inv(E) @ G
C = np.eye(len(A))
D = np.zeros((A.shape[0], B.shape[1])) 

nomes_x = ['$u$ (m/s)', '$\\alpha$ (rad)', '$q$ (rad/s)', '$\\theta$ (rad)', '$\\beta$ (rad)', '$p$ (rad/s)', '$r$ (rad/s)', '$\phi$ (rad)']
nomes_u = ["$\delta_e$ ($^o$)", "$\delta_r$ ($^o$)", "$\delta_a$ ($^o$)"]

x0 = 0.1*np.ones(8)                 # Estado Inicial

t_span = (0, 20)
t_eval = np.linspace(0, 20, 500)
## Análise de estabilidade

eig_MA =    diagrama_polos(A, 'Malha aberta')
sol_MA = resposta_livre(A,  x0, t_span, t_eval, 'Malha aberta')


## Para Controlabilidade e Observabilidade
# Q =  [B | AB | ... | A^n-1B]

Ct =np.hstack([np.linalg.matrix_power(A, n) @ B for n in range(1, len(A)-1)])
Ct = np.concatenate((B, Ct), 1)
rank_Ct = np.linalg.matrix_rank(Ct)

# Ob = [C | CA | ... | C^n-1A].T

Ob =np.vstack([ C @ np.linalg.matrix_power(A, n) for n in range(1, len(A)-1)])
Ob = np.concatenate((C, Ob), 0)
rank_Ob = np.linalg.matrix_rank(Ob)

## LQR
# Ricatti Algébrica: A.t * P + P * A − P * B * R^−1 * B.t * P + Q = 0
# Q = 0.1*np.eye(len(A)) # q * I, com q obtido por teste
# R = 11.1* np.eye(len(B[0])) # r * I, com r obtido por teste

Q = np.diag([ 
    8,   # u (velocidade longitudinal)
    120,   # alpha (ângulo de ataque) 
    0.01,    # q (velocidade angular em arfagem) 
    2000,   # theta (ângulo de arfagem) 
    2000,   # beta (ângulo de derrapagem) 
    0.01,    # p (velocidade angular em rolagem) 
    0.01,    # r (velocidade angular em guinada)
    150   # phi (ângulo de rolamento) 
])

R = np.diag([
    0.08,    # δe (elevador) - principal para controle longitudinal
    0.1,   # δr (leme) - usa-se pouco
    0.01     # δa (aileron) - importante para rolamento
 ]) # Eigvals resultantes: array([-6.93e+02+606.j, -6.93e+02-606.j, -3.48e+02  +0.j, -2.02e+01  +0.j, -3.20e+00 +6.66j, -3.20e+00 -6.66j, -2.28e+00  +0.j, -3.21e-01  +0.j])

K_lqr, P, autovalores_lqr = ctr.lqr(A, B, Q, R)
A_lqr = A - B @ K_lqr

eig_lqr = diagrama_polos(A_lqr, 'LQR')
sol_lqr = resposta_livre(A_lqr,  x0, t_span, t_eval, 'LQR') # Para só os angulos,  ,estados = ['$\\alpha$', '$\\theta$', '$\\beta$','$\phi$']
u_lqr = esforco_atuadores(sol_lqr, K_lqr, 'LQR', plot = True)


# Alocação de polos
polos_alocados = np.array([-20.5+1.07j, -20.5-1.07j, -3.84+3.52j, -3.84-3.52j, -8.83, -1.53,-1.3, -1.25]) # Testes feitos com objetivo de limitar delta_max a 20deg e alfa e beta a 0.2rad
alocacao = place_poles(A, B, polos_alocados) 
K_aloc = alocacao.gain_matrix
A_aloc = A - B @ K_aloc # np.linalg.eigvals(A_aloc)

eig_aloc = diagrama_polos(A_aloc, 'Alocação de Polos')
sol_aloc = resposta_livre(A_aloc,  x0, t_span, t_eval, 'Alocação de Polos') # Para só os angulos,  ,estados = ['$\\alpha$', '$\\theta$', '$\\beta$','$\phi$']
u_aloc = esforco_atuadores(sol_aloc, K_aloc, 'Alocação de Polos', plot = True)

## Observador identidade
polos_obs = 3 * -np.abs(np.real(np.linalg.eigvals(A_lqr)))  # dinamica mais rapida
erro_lqr = observacao(polos_obs, K_lqr, 'LQR', xhat0 = 0.0*np.ones(8))

## Seguidor de referencia com retroalimentação
x_r = np.array([0.1,0.0,0,0.,0,0,0,0]) # Velocidade de referencia 0.1m/s acima do equilibrio natural
sol_seg, u_seg = seguidor_LQR(A, B, K_lqr, Q, R, P, x0, x_r)
