""" WGImplicit

- WG basic + fabric + non-coaxiality
- Storage scheme: stress, strain, fabric as 6X1 vectors
                                                    [a_11, a_22, a_33, sqrt(2)*a_23, sqrt(2)*a_13, sqrt(2)*a_12]
                  constitutive tensor as a 6X6 tensor
                                                    [C_nn, sqrt(2)*C_ns; sqrt(2)*C_sn, 2*C_ss]

DEVELOPED AT:
                    COMPUTATIONAL GEOMECHANICS LABORATORY
                    DEPARTMENT OF CIVIL ENGINEERING
                    UNIVERSITY OF CALGARY, AB, CANADA
                    DIRECTOR: Prof. Richard Wan

DEVELOPED BY:
                    MAHDAD EGHBALIAN

MIT License

Copyright (c) 2022 Mahdad Eghbalian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# import required modules
import numpy as np
import wgimplicit_module_utility as util


class Sample:
    def __init__(self, p0,  mu, nu, g0, vrc0, ncr, hcr, sinphicv, a, beta, alpha, x, chi, hic, nic, cap, a_f, b_0,
                 n_1, n_2, lmbd):
        self.p0 = p0  # reference mean stress (kPa)
        self.mu = mu  # ratio of flow stress under triaxial extension to that under triaxial compression
        self.nu = nu  # Poisson's ratio
        self.g0 = g0  # reference shear modulus (kPa)
        self.vrc0 = vrc0  # critical void ratio at zero confining stress
        self.ncr = ncr  # power constant in the critical void ratio relation
        self.hcr = hcr  # (1.0/0.005)**(1.0/n_cr) constant in the critical void ratio relation
        self.sinphicv = sinphicv  # sin of friction angle at critical state
        self.a = a  # constant in the relation for mobilized friction angle (basic model)
        self.beta = beta  # power constant in the relation for mobilized friction angle
        self.alpha = alpha  # power constant in the relation for friction angle at failure
        self.x = x  # parameter of the fabric dependency for the characteristic friction angle at failure
        self.chi = chi   # parameter of the fabric evolution law
        self.hic = hic  # constant in the cap hardening relation
        self.nic = nic  # power constant in the cap hardening relation
        self.cap = cap  # 1 means cap is considered, 0 means no cap
        self.a_f = a_f  # constant in the relation for mobilized friction angle (fabric model)
        self.b_0 = b_0  # constant in the relation for friction angle at failure (fabric model)
        self.n_1 = n_1  # power in the relation for a_f (fabric model)
        self.n_2 = n_2  # power in the relation for b_f (fabric model)
        self.lmbd = lmbd  # constant of the noncoaxial term in the flow rule
        self.kgratio = 3.0 * (1.0 - 2.0 * nu) / (2 * (1 + nu))

        self.cebar = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i < 3 and j < 3:
                    self.cebar[i][j] += 1.0 - 2.0 * self.kgratio / 3.0

                if i == j:
                    self.cebar[i][j] += 2.0 * self.kgratio


class Load:
    def __init__(self, goal, nstep, strain, stress):
        self.goal = goal  # loading goal; 0: stress controlled, 1: strain controlled
        self.nstep = nstep  # number of solution steps
        self.strain = strain  # total prescribed strain
        self.stress = stress  # total prescribed stress
        self.incre_strain = 0.0
        self.incre_stress = 0.0
        if nstep != 0:
            self.incre_strain = strain / nstep  # incremental prescribed strain
            self.incre_stress = stress / nstep  # incremental prescribed stress

        self.istep = -1


class Solver:
    def __init__(self, tol_rtmp, tol_elas, err_in, maxit_elas, maxit_rtmp, stpinfo, freq, stp_pause):
        self.tol_rtmp = tol_rtmp  # convergence tolerance for the return mapping
        self.tol_elas = tol_elas  # convergence tolerance for the elastic trial step
        self.err_in = err_in  # the initial error to set the iteration loop off
        self.maxit_elas = maxit_elas  # maximum number of iterations for the elastic trial step
        self.maxit_rtmp = maxit_rtmp  # maximum number of iterations for the return mapping
        self.stpinfo = stpinfo  # True(False) to print or not print the step info
        self.freq = freq  # if stepinfo is 'false', at least print step number with this frequency
        self.stp_pause = stp_pause  # pause before going to next step (for debugging)


class State:
    def __init__(self, initial_value):
        self.value = initial_value
        self.value_iter = initial_value
        self.value_step = initial_value
        self.value_trial = initial_value
        self.value_in = initial_value

    def update(self, dvalue):
        self.value = self.value + dvalue

    def set_iter(self):
        self.value_iter = self.value

    def set_step(self):
        self.value_step = self.value

    def set_trial(self):
        self.value_trial = self.value

    def reset(self):
        self.value = self.value_trial
        self.value_iter = self.value_trial


class StateScalar(State):
    def error(self):
        if abs(self.value) > 1e-10:
            error = (self.value - self.value_iter) / self.value
        else:
            error = 0.0
        return error


class StateVector(State):
    def error(self):
        norm_val = util.norm(self.value)
        norm_pre = util.norm(self.value_iter)
        norm_in = util.norm(self.value_in)
        if abs(norm_val - norm_in) > 1e-10:
            error = (norm_val - norm_pre) / (norm_val - norm_in)
        else:
            error = 0.0
        return error


class Solution:
    def __init__(self, strain, stress, strain_pl, strain_pl_c, vr, f):
        self.strain = np.transpose(strain.value)
        self.stress = np.transpose(stress.value)
        self.strain_pl = np.transpose(strain_pl.value)
        self.strain_pl_c = np.transpose(strain_pl_c.value)
        self.vr = [[vr.value]]
        self.ksecant = [[0.0]]
        self.sphm = [[0.0]]
        self.mmf = [[0.0]]
        self.f = np.transpose(f.value)
        self.p = [[util.mean(stress.value)]]
        self.q = [[util.eq(stress.value)]]
        strain_pl_j2 = util.j2(strain_pl.value)
        self.gamma_pl = [[np.sqrt((4.0 / 3.0) * strain_pl_j2)]]
        strain_pl_star = util.starf(strain_pl.value, f)
        strain_pl_j2_star = util.j2(strain_pl_star)
        self.gamma_pl_star = [[np.sqrt((4.0 / 3.0) * strain_pl_j2_star)]]
        strain_j2 = util.j2(strain.value)
        self.gamma = [[np.sqrt((4.0 / 3.0) * strain_j2)]]
        self.axial_strain = [[strain.value[2][0] - strain.value_in[2][0]]]
        self.axial_stress = [[stress.value[2][0]]]

    def record(self, strain, stress, strain_pl, strain_pl_c, vr, f, sphm, mmf, ksecant):
        self.strain = np.append(self.strain, np.transpose(strain.value), axis=0)
        self.stress = np.append(self.stress, np.transpose(stress.value), axis=0)
        self.strain_pl = np.append(self.strain_pl, np.transpose(strain_pl.value), axis=0)
        self.strain_pl_c = np.append(self.strain_pl_c, np.transpose(strain_pl_c.value), axis=0)
        self.f = np.append(self.f, np.transpose(f.value), axis=0)
        self.vr = np.append(self.vr, [[vr.value]], axis=0)
        self.ksecant = np.append(self.ksecant, [[ksecant]], axis=0)
        self.sphm = np.append(self.sphm, [[sphm]], axis=0)
        self.mmf = np.append(self.mmf, [[mmf]], axis=0)

        p = util.mean(stress.value)
        q = util.eq(stress.value)
        strain_pl_j2 = util.j2(strain_pl.value)
        gamma_pl = np.sqrt((4.0 / 3.0) * strain_pl_j2)
        strain_pl_star = util.starf(strain_pl.value, f)
        strain_pl_j2_star = util.j2(strain_pl_star)
        gamma_pl_star = np.sqrt((4.0 / 3.0) * strain_pl_j2_star)
        strain_j2 = util.j2(strain.value)
        gamma = np.sqrt((4.0 / 3.0) * strain_j2)
        axial_strain = strain.value[2][0]
        axial_stress = stress.value[2][0]
        self.p = np.append(self.p, [[p]], axis=0)
        self.q = np.append(self.q, [[q]], axis=0)
        self.gamma_pl = np.append(self.gamma_pl, [[gamma_pl]], axis=0)
        self.gamma_pl_star = np.append(self.gamma_pl_star, [[gamma_pl_star]], axis=0)
        self.gamma = np.append(self.gamma, [[gamma]], axis=0)
        self.axial_strain = np.append(self.axial_strain, [[axial_strain]], axis=0)
        self.axial_stress = np.append(self.axial_stress, [[axial_stress]], axis=0)


class Mechanism:
    def __init__(self):
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False
