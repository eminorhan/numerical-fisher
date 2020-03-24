'''Defines the problem classes'''
import numpy as np
from scipy.special import iv
from scipy.optimize import minimize
import matplotlib as mp
import matplotlib.pylab as plt


TWOPI = 2.0 * np.pi


class PopulationCode:
    '''Exact population code solution with no constraints'''
    def __init__(self, nneurons=100, ncomps=4, nstims=500, kappa_comp=2.0, kappa_prior=2.0, mu_prior=np.pi, R=1.):
        self.nneurons = nneurons
        self.ncomps = ncomps
        self.nstims = nstims
        self.kappa_comp = kappa_comp
        self.kappa_prior = kappa_prior
        self.mu_prior = mu_prior
        self.R = R

        self.delta_phi = (TWOPI / self.nneurons) * np.ones(self.nneurons)
        self.unif_range = np.linspace(0, TWOPI, self.nneurons)
        self.phi = np.repeat(np.expand_dims(self.unif_range, axis=1), self.ncomps, axis=1)
        self.s_range = np.linspace(0, TWOPI, self.nstims)
        self.s_rep = np.repeat(np.expand_dims(self.s_range, axis=1), self.nneurons, axis=1)

        self.prior = np.exp(self.kappa_prior * np.cos(self.s_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))

        self.g_weights = 0.1 * np.random.rand(self.ncomps)
        self.g_centers = TWOPI * np.random.rand(self.ncomps)

        self.w_weights = 0.1 * np.random.rand(self.ncomps)
        self.w_centers = TWOPI * np.random.rand(self.ncomps)

        self.q_weights = 0.1 * np.random.rand(self.ncomps)
        self.q_centers = TWOPI * np.random.rand(self.ncomps)

        self.params = np.concatenate((self.g_weights, self.g_centers, self.w_weights,
                                      self.w_centers, self.q_weights, self.q_centers))

    def unpack_params(self, params):
        g_weights = params[:self.ncomps]
        g_centers = params[self.ncomps:2*self.ncomps]
        w_weights = params[2*self.ncomps:3*self.ncomps]
        w_centers = params[3*self.ncomps:4*self.ncomps]
        q_weights = params[4*self.ncomps:5*self.ncomps]
        q_centers = params[5*self.ncomps:]

        return g_weights, g_centers, w_weights, w_centers, q_weights, q_centers

    def compute_all(self, params):
        g_weights, g_centers, w_weights, w_centers, q_weights, q_centers = self.unpack_params(params)
        g = np.sum(g_weights * np.exp(self.kappa_comp * (np.cos(self.phi - g_centers) - 1)), axis=1)
        w = np.sum(w_weights * np.exp(self.kappa_comp * (np.cos(self.phi - w_centers) - 1)), axis=1)
        q = np.sum(q_weights * np.exp(self.kappa_comp * (np.cos(self.phi - q_centers) - 1)), axis=1)
        q = q / np.sum(q)  # normalize q

        f = g * np.exp(w * (np.cos(self.s_rep - TWOPI * np.cumsum(q)) - 1))
        f_prime = - w * f * np.sin(self.s_rep - TWOPI * np.cumsum(q))
        I_f = np.sum(f_prime ** 2 / f, axis=1)

        return g, w, q, f, f_prime, I_f

    def loss(self, params):
        _, _, _, _, _, I_f = self.compute_all(params)
        return - (TWOPI / self.nstims) * np.sum(self.prior * np.log(I_f))  # loss

    def eq_constraint(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return (TWOPI / self.nstims) * np.sum(self.prior * np.sum(f, axis=1)) - self.R

    def ineq_constraint_f(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return np.min(f)

    def ineq_constraint_w(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return np.min(w)

    def ineq_constraint_w2(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return - np.max(w) + 8

    def ineq_constraint_g(self, params):
        g, _, _, _, _, _ = self.compute_all(params)
        return np.min(g)

    def ineq_constraint_q(self, params):
        _, _, q, _, _, _ = self.compute_all(params)
        return np.min(q)

    def loss_callback(self, params):
        print(self.loss(params))

    def optimize_fisher(self):
        cons = ({'type': 'eq', 'fun': self.eq_constraint},
                {'type': 'ineq', 'fun': self.ineq_constraint_f},
                {'type': 'ineq', 'fun': self.ineq_constraint_g},
                {'type': 'ineq', 'fun': self.ineq_constraint_w},
                # {'type': 'ineq', 'fun': self.ineq_constraint_w2},
                {'type': 'ineq', 'fun': self.ineq_constraint_q}
                )

        opt = minimize(self.loss, self.params,
                       callback=self.loss_callback,
                       constraints=cons,
                       options={'maxiter': 1000, 'disp': True})

        self.params = opt.x
        return self.params

    def plot_results(self, params):
        g, w, q, f, f_prime, I_f = self.compute_all(params)
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.unif_range, g)
        plt.plot([np.pi, np.pi], [0, .001], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(g)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Gain', fontsize=10)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        ax2 = plt.subplot(2, 2, 2)
        plt.plot(self.unif_range, w)
        plt.plot([np.pi, np.pi], [0, np.max(w)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(w)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Precision', fontsize=10)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')

        ax3 = plt.subplot(2, 2, 3)
        plt.plot(self.unif_range, 1/q)
        plt.plot([np.pi, np.pi], [0, np.max(1/q)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(1/q)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Density', fontsize=10)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')

        ax4 = plt.subplot(2, 2, 4)
        plt.plot(f)
        plt.plot(self.prior)
        #plt.plot([np.pi, np.pi], [0, np.max(f)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        # plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(f)])
        plt.ylabel('Tuning funcs.', fontsize=10)
        ax4.spines["right"].set_visible(False)
        ax4.spines["top"].set_visible(False)
        ax4.yaxis.set_ticks_position('left')
        ax4.xaxis.set_ticks_position('bottom')

        plt.tight_layout()

        mp.rcParams['axes.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 1.15
        mp.rcParams['font.sans-serif'] = ['FreeSans']
        mp.rcParams['mathtext.fontset'] = 'cm'
        plt.savefig('optims.pdf', bbox_inches='tight')


class PopulationCodeV2:
    '''Exact population code solution with no constraints'''
    def __init__(self, nneurons=100, ncomps=4, nstims=500, kappa_comp=2.0, kappa_prior=2.0, mu_prior=np.pi, R=1.):
        self.nneurons = nneurons
        self.ncomps = ncomps
        self.nstims = nstims
        self.kappa_comp = kappa_comp
        self.kappa_prior = kappa_prior
        self.mu_prior = mu_prior
        self.R = R

        self.delta_phi = (TWOPI / self.nneurons) * np.ones(self.nneurons)
        self.unif_range = np.linspace(0, TWOPI, self.nneurons)
        self.phi = np.repeat(np.expand_dims(self.unif_range, axis=1), self.ncomps, axis=1)
        self.s_range = np.linspace(0, TWOPI, self.nstims)
        self.s_rep = np.repeat(np.expand_dims(self.s_range, axis=1), self.nneurons, axis=1)
        self.s_w = np.repeat(np.expand_dims(self.s_range, axis=1), self.ncomps, axis=1)

        self.prior = np.exp(self.kappa_prior * np.cos(self.s_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))

        self.g_weights = 0.1 * np.random.rand(self.ncomps)
        self.g_centers = TWOPI * np.random.rand(self.ncomps)

        self.w_weights = 0.1 * np.random.rand(self.ncomps)
        self.w_centers = TWOPI * np.random.rand(self.ncomps)

        self.q_weights = 0.1 * np.random.rand(self.ncomps)
        self.q_centers = TWOPI * np.random.rand(self.ncomps)

        self.params = np.concatenate((self.g_weights, self.g_centers, self.w_weights,
                                      self.w_centers, self.q_weights, self.q_centers))

    def unpack_params(self, params):
        g_weights = params[:self.ncomps]
        g_centers = params[self.ncomps:2*self.ncomps]
        w_weights = params[2*self.ncomps:3*self.ncomps]
        w_centers = params[3*self.ncomps:4*self.ncomps]
        q_weights = params[4*self.ncomps:5*self.ncomps]
        q_centers = params[5*self.ncomps:]

        return g_weights, g_centers, w_weights, w_centers, q_weights, q_centers

    def compute_all(self, params):
        g_weights, g_centers, w_weights, w_centers, q_weights, q_centers = self.unpack_params(params)
        g = np.sum(g_weights * np.exp(self.kappa_comp * (np.cos(self.phi - g_centers) - 1)), axis=1)
        q = np.sum(q_weights * np.exp(self.kappa_comp * (np.cos(self.phi - q_centers) - 1)), axis=1)
        q = q / np.sum(q)  # normalize q

        # w is a function of s
        w = np.sum(w_weights * np.exp(self.kappa_comp * (np.cos(self.s_w - w_centers) - 1)), axis=1)
        w_prime = - np.sum(w_weights * np.exp(self.kappa_comp * (np.cos(self.s_w - w_centers) - 1)) *
                           np.sin(self.s_w - w_centers), axis=1)

        f = g * np.exp(np.expand_dims(w, axis=1) * (np.cos(self.s_rep - TWOPI * np.cumsum(q)) - 1))
        f_prime = f * (np.expand_dims(w_prime, axis=1) * (np.cos(self.s_rep - TWOPI * np.cumsum(q)) - 1) -
                       np.expand_dims(w, axis=1) * np.sin(self.s_rep - TWOPI * np.cumsum(q)))
        I_f = np.sum(f_prime ** 2 / f, axis=1)

        return g, w, q, f, f_prime, I_f

    def loss(self, params):
        _, _, _, _, _, I_f = self.compute_all(params)
        return - (TWOPI / self.nstims) * np.sum(self.prior * np.log(I_f))  # loss

    def eq_constraint(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return (TWOPI / self.nstims) * np.sum(self.prior * np.sum(f, axis=1)) - self.R

    def ineq_constraint_f(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return np.min(f)

    def ineq_constraint_w(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return np.min(w)

    def ineq_constraint_w2(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return - np.max(w) + 8

    def ineq_constraint_g(self, params):
        g, _, _, _, _, _ = self.compute_all(params)
        return np.min(g)

    def ineq_constraint_q(self, params):
        _, _, q, _, _, _ = self.compute_all(params)
        return np.min(q)

    def loss_callback(self, params):
        print(self.loss(params))

    def optimize_fisher(self):
        cons = ({'type': 'eq', 'fun': self.eq_constraint},
                {'type': 'ineq', 'fun': self.ineq_constraint_f},
                {'type': 'ineq', 'fun': self.ineq_constraint_g},
                {'type': 'ineq', 'fun': self.ineq_constraint_w},
                # {'type': 'ineq', 'fun': self.ineq_constraint_w2},
                {'type': 'ineq', 'fun': self.ineq_constraint_q}
                )

        opt = minimize(self.loss, self.params,
                       callback=self.loss_callback,
                       constraints=cons,
                       options={'maxiter': 1000, 'disp': True})

        self.params = opt.x
        return self.params

    def plot_results(self, params):
        g, w, q, f, f_prime, I_f = self.compute_all(params)
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.unif_range, g)
        plt.plot([np.pi, np.pi], [0, .001], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(g)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Gain', fontsize=10)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        ax2 = plt.subplot(2, 2, 2)
        plt.plot(self.s_range, w)
        plt.plot([np.pi, np.pi], [0, np.max(w)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(w)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Precision', fontsize=10)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')

        ax3 = plt.subplot(2, 2, 3)
        plt.plot(self.unif_range, 1/q)
        plt.plot([np.pi, np.pi], [0, np.max(1/q)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(1/q)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Density', fontsize=10)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')

        ax4 = plt.subplot(2, 2, 4)
        plt.plot(f)
        plt.plot(self.prior)
        #plt.plot([np.pi, np.pi], [0, np.max(f)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        # plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(f)])
        plt.ylabel('Tuning funcs.', fontsize=10)
        ax4.spines["right"].set_visible(False)
        ax4.spines["top"].set_visible(False)
        ax4.yaxis.set_ticks_position('left')
        ax4.xaxis.set_ticks_position('bottom')

        plt.tight_layout()

        mp.rcParams['axes.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 1.15
        mp.rcParams['font.sans-serif'] = ['FreeSans']
        mp.rcParams['mathtext.fontset'] = 'cm'
        plt.savefig('optimsV2.pdf', bbox_inches='tight')


class GSPopulationCode:

    def __init__(self, nneurons=100, ncomps=4, nstims=500, kappa_comp=2.0, kappa_prior=2.0, mu_prior=np.pi, R=1.):
        self.nneurons = nneurons
        self.ncomps = ncomps
        self.nstims = nstims
        self.kappa_comp = kappa_comp
        self.kappa_prior = kappa_prior
        self.mu_prior = mu_prior
        self.R = R

        self.delta_phi = (TWOPI / self.nneurons) * np.ones(self.nneurons)
        self.unif_range = np.linspace(0, TWOPI, self.nneurons)
        self.phi = np.repeat(np.expand_dims(self.unif_range, axis=1), self.ncomps, axis=1)
        self.s_range = np.linspace(0, TWOPI, self.nstims)
        self.s_rep = np.repeat(np.expand_dims(self.s_range, axis=1), self.nneurons, axis=1)

        self.prior = np.exp(self.kappa_prior * np.cos(self.s_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))

        self.g_weights = 1. * np.random.rand(self.ncomps)
        self.g_centers = TWOPI * np.random.rand(self.ncomps)

        self.q_weights = 10. * np.random.rand(self.ncomps)
        self.q_centers = TWOPI * np.random.rand(self.ncomps)

        self.params = np.concatenate((self.g_weights, self.g_centers, self.q_weights, self.q_centers))

    def unpack_params(self, params):
        g_weights = params[:self.ncomps]
        g_centers = params[self.ncomps:2*self.ncomps]
        q_weights = params[2*self.ncomps:3*self.ncomps]
        q_centers = params[3*self.ncomps:]

        return g_weights, g_centers, q_weights, q_centers

    def compute_all(self, params):
        g_weights, g_centers, q_weights, q_centers = self.unpack_params(params)

        g = np.sum(g_weights * np.exp(self.kappa_comp * (np.cos(self.phi - g_centers) - 1)), axis=1)
        q0 = np.sum(q_weights * np.exp(self.kappa_comp * (np.cos(self.phi - q_centers) - 1)) / (TWOPI * iv(0, self.kappa_comp)), axis=1)
        q = q0 / np.sum(q0)  # normalize q
        w = 2 * (1 / q0) ** 2

        f = g * np.exp(w * (np.cos(self.s_rep - TWOPI * np.cumsum(q)) - 1))
        f_prime = - w * f * np.sin(self.s_rep - TWOPI * np.cumsum(q))
        I_f = np.sum(f_prime ** 2 / f, axis=1)

        return g, w, q, f, f_prime, I_f

    def loss(self, params):
        _, _, _, _, _, I_f = self.compute_all(params)
        return - (TWOPI / self.nstims) * np.sum(self.prior * np.log(I_f))  # loss

    def eq_constraint(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return (TWOPI / self.nstims) * np.sum(self.prior * np.sum(f, axis=1)) - self.R

    def ineq_constraint_f(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return np.min(f)

    def ineq_constraint_g(self, params):
        g, _, _, _, _, _ = self.compute_all(params)
        return np.min(g)

    def ineq_constraint_q(self, params):
        _, _, q, _, _, _ = self.compute_all(params)
        return np.min(q)

    def loss_callback(self, params):
        print(self.loss(params))

    def optimize_fisher(self):
        cons = ({'type': 'eq', 'fun': self.eq_constraint},
                {'type': 'ineq', 'fun': self.ineq_constraint_f},
                {'type': 'ineq', 'fun': self.ineq_constraint_g},
                {'type': 'ineq', 'fun': self.ineq_constraint_q},
                )

        opt = minimize(self.loss, self.params,
                       callback=self.loss_callback,
                       constraints=cons,
                       options={'maxiter': 1000, 'disp': True})

        self.params = opt.x
        return self.params

    def plot_results(self, params):
        g, w, q, f, f_prime, I_f = self.compute_all(params)
        ax1 = plt.subplot(2,2,1)
        plt.plot(self.unif_range, g)
        plt.plot([np.pi, np.pi], [0, .001], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(g)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Gain', fontsize=10)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        ax2 = plt.subplot(2,2,2)
        plt.plot(self.unif_range, w)
        plt.plot([np.pi, np.pi], [0, np.max(w)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(w)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Precision', fontsize=10)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')

        ax3 = plt.subplot(2,2,3)
        plt.plot(self.unif_range, 1/q)
        plt.plot([np.pi, np.pi], [0, np.max(1/q)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(1/q)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Density', fontsize=10)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')

        ax4 = plt.subplot(2,2,4)
        plt.plot(f)
        plt.plot(self.prior)
        #plt.plot([np.pi, np.pi], [0, np.max(f)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        # plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(f)])
        plt.ylabel('Tuning funcs.', fontsize=10)
        ax4.spines["right"].set_visible(False)
        ax4.spines["top"].set_visible(False)
        ax4.yaxis.set_ticks_position('left')
        ax4.xaxis.set_ticks_position('bottom')

        plt.tight_layout()

        mp.rcParams['axes.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 1.15
        mp.rcParams['font.sans-serif'] = ['FreeSans']
        mp.rcParams['mathtext.fontset'] = 'cm'
        plt.savefig('gs_optims.pdf', bbox_inches='tight')


class GSPopulationCodeV2:
    '''Exact population code solution with no constraints'''
    def __init__(self, nneurons=100, ncomps=4, nstims=500, kappa_comp=2.0, kappa_prior=2.0, mu_prior=np.pi, R=1.):
        self.nneurons = nneurons
        self.ncomps = ncomps
        self.nstims = nstims
        self.kappa_comp = kappa_comp
        self.kappa_prior = kappa_prior
        self.mu_prior = mu_prior
        self.R = R

        self.delta_phi = (TWOPI / self.nneurons) * np.ones(self.nneurons)
        self.unif_range = np.linspace(0, TWOPI, self.nneurons)
        self.phi = np.repeat(np.expand_dims(self.unif_range, axis=1), self.ncomps, axis=1)
        self.s_range = np.linspace(0, TWOPI, self.nstims)
        self.s_rep = np.repeat(np.expand_dims(self.s_range, axis=1), self.nneurons, axis=1)
        self.s_w = np.repeat(np.expand_dims(self.s_range, axis=1), self.ncomps, axis=1)

        self.prior = np.exp(self.kappa_prior * np.cos(self.s_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))

        self.g_weights = 0.1 * np.random.rand(self.ncomps)
        self.g_centers = TWOPI * np.random.rand(self.ncomps)

        self.w_weights = 0.1 * np.random.rand(self.ncomps)
        self.w_centers = TWOPI * np.random.rand(self.ncomps)

        self.q_weights = 0.1 * np.random.rand(self.ncomps)
        self.q_centers = TWOPI * np.random.rand(self.ncomps)

        self.params = np.concatenate((self.g_weights, self.g_centers, self.w_weights,
                                      self.w_centers, self.q_weights, self.q_centers))

    def unpack_params(self, params):
        g_weights = params[:self.ncomps]
        g_centers = params[self.ncomps:2*self.ncomps]
        w_weights = params[2*self.ncomps:3*self.ncomps]
        w_centers = params[3*self.ncomps:4*self.ncomps]
        q_weights = params[4*self.ncomps:5*self.ncomps]
        q_centers = params[5*self.ncomps:]

        return g_weights, g_centers, w_weights, w_centers, q_weights, q_centers

    def compute_all(self, params):
        g_weights, g_centers, w_weights, w_centers, q_weights, q_centers = self.unpack_params(params)
        g = np.sum(g_weights * np.exp(self.kappa_comp * (np.cos(self.phi - g_centers) - 1)), axis=1)

        # w is a function of s
        w = np.sum(w_weights * np.exp(self.kappa_comp * (np.cos(self.s_w - w_centers) - 1)), axis=1)
        w_prime = - np.sum(w_weights * np.exp(self.kappa_comp * (np.cos(self.s_w - w_centers) - 1)) *
                           np.sin(self.s_w - w_centers), axis=1)

        q = w / np.sum(w)  # normalize q
        q = 2 * (1 / q) ** 2

        f = g * np.exp(np.expand_dims(w, axis=1) * (np.cos(self.s_rep - TWOPI * np.expand_dims(np.cumsum(q), axis=1) ) - 1))
        f_prime = f * (np.expand_dims(w_prime, axis=1) * (np.cos(self.s_rep - TWOPI * np.expand_dims(np.cumsum(q), axis=1) ) - 1) -
                       np.expand_dims(w, axis=1) * np.sin(self.s_rep - TWOPI * np.expand_dims(np.cumsum(q), axis=1) ))
        I_f = np.sum(f_prime ** 2 / f, axis=1)

        return g, w, q, f, f_prime, I_f

    def loss(self, params):
        _, _, _, _, _, I_f = self.compute_all(params)
        return - (TWOPI / self.nstims) * np.sum(self.prior * np.log(I_f))  # loss

    def eq_constraint(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return (TWOPI / self.nstims) * np.sum(self.prior * np.sum(f, axis=1)) - self.R

    def ineq_constraint_f(self, params):
        _, _, _, f, _, _ = self.compute_all(params)
        return np.min(f)

    def ineq_constraint_w(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return np.min(w)

    def ineq_constraint_w2(self, params):
        _, w, _, _, _, _ = self.compute_all(params)
        return - np.max(w) + 8

    def ineq_constraint_g(self, params):
        g, _, _, _, _, _ = self.compute_all(params)
        return np.min(g)

    def ineq_constraint_q(self, params):
        _, _, q, _, _, _ = self.compute_all(params)
        return np.min(q)

    def loss_callback(self, params):
        print(self.loss(params))

    def optimize_fisher(self):
        cons = ({'type': 'eq', 'fun': self.eq_constraint},
                {'type': 'ineq', 'fun': self.ineq_constraint_f},
                {'type': 'ineq', 'fun': self.ineq_constraint_g},
                {'type': 'ineq', 'fun': self.ineq_constraint_w},
                # {'type': 'ineq', 'fun': self.ineq_constraint_w2},
                {'type': 'ineq', 'fun': self.ineq_constraint_q}
                )

        opt = minimize(self.loss, self.params,
                       callback=self.loss_callback,
                       constraints=cons,
                       options={'maxiter': 1000, 'disp': True})

        self.params = opt.x
        return self.params

    def plot_results(self, params):
        g, w, q, f, f_prime, I_f = self.compute_all(params)
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.unif_range, g)
        plt.plot([np.pi, np.pi], [0, .001], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(g)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Gain', fontsize=10)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        ax2 = plt.subplot(2, 2, 2)
        plt.plot(self.s_range, w)
        plt.plot([np.pi, np.pi], [0, np.max(w)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(w)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Precision', fontsize=10)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')

        ax3 = plt.subplot(2, 2, 3)
        plt.plot(self.s_range, 1/q)
        plt.plot([np.pi, np.pi], [0, np.max(1/q)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(1/q)])
        plt.xticks([0, np.pi, TWOPI], ['0', '$\pi$', '2$\pi$'])
        plt.ylabel('Density', fontsize=10)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')

        ax4 = plt.subplot(2, 2, 4)
        plt.plot(f)
        plt.plot(self.prior)
        #plt.plot([np.pi, np.pi], [0, np.max(f)], '-', color=[.5, .5, .5])
        plt.tick_params(labelsize=9, direction='out')
        # plt.xlim([0, TWOPI])
        plt.ylim([0, np.max(f)])
        plt.ylabel('Tuning funcs.', fontsize=10)
        ax4.spines["right"].set_visible(False)
        ax4.spines["top"].set_visible(False)
        ax4.yaxis.set_ticks_position('left')
        ax4.xaxis.set_ticks_position('bottom')

        plt.tight_layout()

        mp.rcParams['axes.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 1.15
        mp.rcParams['font.sans-serif'] = ['FreeSans']
        mp.rcParams['mathtext.fontset'] = 'cm'
        plt.savefig('GSoptimsV2.pdf', bbox_inches='tight')


class GSPopulationCodeExact:
    '''Exact population code solution'''
    def __init__(self, nneurons=100, nstims=500, kappa_prior=2.0, mu_prior=np.pi, R=1.):
        self.nneurons = nneurons
        self.nstims = nstims
        self.kappa_prior = kappa_prior
        self.mu_prior = mu_prior
        self.R = R

        self.delta_phi = (TWOPI / self.nneurons) * np.ones(self.nneurons)
        self.unif_range = np.linspace(0, TWOPI, self.nneurons)
        self.s_range = np.linspace(0, TWOPI, self.nstims)
        self.s_rep = np.repeat(np.expand_dims(self.s_range, axis=1), self.nneurons, axis=1)

        self.prior = np.exp(self.kappa_prior * np.cos(self.s_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))
        self.prior_n = np.exp(self.kappa_prior * np.cos(self.unif_range - self.mu_prior)) / (TWOPI * iv(0, self.kappa_prior))

        self.g = self.R * np.ones(self.nneurons) / self.nneurons
        self.q = 1 / self.prior_n
        self.q = self.q / np.sum(self.q)
        self.w = (self.prior_n *self.nneurons) ** 2

        self.f = self.g * np.exp(self.w * (np.cos(self.s_rep - TWOPI * np.cumsum(self.q)) - 1) )
        self.f_prime = - self.w * self.f * np.sin(self.s_rep - TWOPI * np.cumsum(self.q))
        self.I_f = np.sum(self.f_prime ** 2 / self.f, axis=1)
        self.loss = - (TWOPI / self.nstims) * np.sum(self.prior * np.log(self.I_f))  # loss

    def eq_constraint(self):
        return (TWOPI / self.nstims) * np.sum(self.prior * np.sum(self.f, axis=1)) - self.R

    def ineq_constraint_f(self):
        return np.min(self.f)

    def ineq_constraint_g(self):
        return np.min(self.g)

    def ineq_constraint_q(self):
        return np.min(self.q)
