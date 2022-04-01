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
from wgimplicit_classes import StateScalar, StateVector, Mechanism
import wgimplicit_module_utility as util
import sys
import time
import pickle
import numpy as np
import math


def elastic_iteration(func_arg, criteria, iteration):
    smp, stress, vr, load = func_arg

    p = util.mean(stress.value)
    dstrain_vol = criteria

    if dstrain_vol.value == 0.0:
        ksec = -(smp.g0 * np.sqrt(smp.p0) * np.sqrt(p) * (4.7089 + (-4.34 + vr.value) * vr.value) *
                 (1.0 + smp.nu)) / (3.0 * (1.0 + vr.value) * (-0.5 + smp.nu))
    else:
        ksec = (-p + (np.sqrt(p) - (smp.g0 * np.sqrt(smp.p0) * (1.0 + smp.nu) * (-2.17 + vr.value - 10.0489 /
                                                                                 (1.0 + vr.value) -
                                                                                 6.34 * np.log(1.0 + vr.value))) /
                      (3.0 * (-1.0 + 2.0 * smp.nu)) + (smp.g0 * np.sqrt(smp.p0) * (1.0 + smp.nu) *
                                                       (-2.17 + vr.value + 10.0489 / ((1.0 + vr.value) *
                                                                                      (-1.0 + dstrain_vol.value)) -
                                                        (1.0 + vr.value) * dstrain_vol.value - 6.34 *
                                                        np.log(-(1.0 + vr.value) * (-1.0 + dstrain_vol.value)))) /
                      (3.0 * (-1.0 + 2.0 * smp.nu))) ** 2.0) / dstrain_vol.value

    ce = ksec * smp.cebar

    dstress = StateVector(initial_value=np.zeros((6, 1)))
    dstrain = StateVector(initial_value=np.zeros((6, 1)))

    key = []
    for i in range(5, -1, -1):
        if load.goal[i] == 1:
            key.append(i)
        else:
            dstress.value[i][0] = load.incre_stress[i][0]

    cecc = ce
    for i in key:
        cec = np.delete(cecc, i, 0)
        cecc = np.delete(cec, i, 1)
        dstrain.value[i][0] = load.incre_strain[i][0]

    dstress.update(-np.matmul(ce, dstrain.value))
    dstressc = dstress.value
    for i in key:
        dstressc = np.delete(dstressc, i, 0)

    dstrainc = np.matmul(np.linalg.inv(cecc), dstressc)

    dstrain.value = dstrainc
    for i in reversed(key):
        dstrain.value = np.insert(dstrain.value, i, load.incre_strain[i][0], axis=0)

    dstress.value = np.matmul(ce, dstrain.value)

    dp = util.mean(dstress.value)
    dstrain_vol.value = dp / ksec
    error = abs(dstrain_vol.error())

    dstrain_vol.set_iter()

    return [error, dstrain_vol, dstress, dstrain, ksec]


def iterator(func, func_arg, toler, error_in, criteria, maxiter):

    flag = 0
    error_flag = 0
    iteration = 0
    func_out = []
    error_1 = error_in
    kount_error = 0
    while error_1 > toler:
        iteration += 1
        func_out = func(func_arg, criteria, iteration)
        error_2 = func_out[0]
        criteria = func_out[1]

        if error_2 > error_1:
            kount_error += 1

        error_1 = error_2
        if kount_error >= 5:
            error_flag = 1
            break

        if iteration == maxiter:
            flag = 1
            break

    return func_out, flag, iteration, error_flag


def state_update(elastic_out, vr, stress, strain, strain_pl, strain_pl_c, f, smp):

    ksec = elastic_out[4]
    delta_eps_vol = 3.0 * util.mean(elastic_out[3].value)
    vr.update((1.0 + vr.value) * np.exp(-delta_eps_vol) - 1.0 - vr.value)
    vr.set_iter()
    vr.set_trial()
    stress.update(elastic_out[2].value)
    stress.set_iter()
    stress.set_trial()
    strain.update(elastic_out[3].value)
    strain.set_iter()
    strain.set_trial()
    strain_pl.set_iter()
    strain_pl.set_trial()
    strain_pl_c.set_iter()
    strain_pl_c.set_trial()

    p = util.mean(stress.value)
    p_step = util.mean(stress.value_step)
    deltaf = np.zeros((6, 1))
    for i in range(6):
        deltaf[i][0] = smp.chi * (p * (stress.value[i][0] - stress.value_step[i][0]) -
                                  (p - p_step) * stress.value_step[i][0]) / p ** 2.0

    f.update(deltaf)
    f.set_iter()
    f.set_trial()

    return ksec, stress, strain, vr, f, strain_pl, strain_pl_c


def elastic_trial(slvr, smp, vr, load, stress, istep, strain, strain_pl, strain_pl_c, f):

    dstrain_vol = StateScalar(initial_value=0.0)
    elastic_out, conflag, iteration, error_flag = iterator(func=elastic_iteration, func_arg=[smp, stress, vr, load],
                                                           toler=slvr.tol_elas, error_in=slvr.err_in,
                                                           criteria=dstrain_vol, maxiter=slvr.maxit_elas)
    if conflag == 1:
        print("Maxiumum iterations reached in elastic trial at step " + str(istep))
        sys.exit()
    elif error_flag == 1:
        print('error is increasing in 5 consecutive iterations in elastic trial step')
        sys.exit()
    else:
        if slvr.stpinfo:
            print("elastic trial converged with " + str(iteration) + " iterations")

    ksec, stress, strain, vr, f, strain_pl, strain_pl_c = state_update(elastic_out, vr, stress, strain, strain_pl,
                                                                       strain_pl_c, f, smp)

    return ksec, stress, strain, vr, f, strain_pl, strain_pl_c


def check_trial(stress, strain_pl, smp, vr, strain_pl_c, f):

    ifplastic = False
    shear = Mechanism()
    cap = Mechanism()

    yield_shear_trial = yield_shear_fabric(strain_pl, smp, vr, f, stress)
    yield_cap_trial = yield_cap(stress, strain_pl_c, smp, f)

    q = util.eq(stress.value)

    if smp.cap == 0 and q < 1e-08:
        pass
    elif smp.cap == 0:
        if yield_shear_trial > 0:
            ifplastic = True
            shear.activate()
    elif q < 1e-08:
        if yield_cap_trial > 0:
            ifplastic = True
            cap.activate()
    else:
        if yield_shear_trial > 0 or yield_cap_trial > 0:
            ifplastic = True
            shear.activate()
            cap.activate()

    return ifplastic, shear, cap


def accept_trial(stress, strain, vr, f):

    stress.set_step()
    strain.set_step()
    vr.set_step()
    f.set_step()

    return stress, strain, vr, f


def get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c, delta_lambda_s, delta_lambda_c, load,
                  strain, istep, shear, cap, f):
    # ===  initial calculations:

    # constants
    _, _, _, sinpsim_f, _, _, a_f = yield_params(smp, vr, stress, strain_pl, f)

    key_strain = []
    key_stress = []
    for i in range(6):
        if load.goal[i] == 1:
            key_strain.append(i)
        else:
            key_stress.append(i)

    sq0 = np.zeros((6, 1))
    delta = np.zeros((6, 1))
    for i in range(6):
        sq0[i][0] = np.sqrt(2.0 / 3.0)
        if i < 3:
            delta[i][0] = 1
            sq0[i][0] = 2.0 / 3.0

    # stress_1
    stress_star = util.starinvf(stress.value, f)
    stress_star_step = util.starinvf(stress.value_step, f)
    stress_star_inv_stress_n = util.invstarinvfn(stress, f)
    stress_star_inv = util.invstarinvf(stress.value, f)
    del_stress_total = np.matmul(ksec * smp.cebar, (strain.value - strain.value_step))
    del_stress_pl = np.matmul(ksec * smp.cebar, (strain_pl.value - strain_pl.value_step))

    # stress_2
    s = util.deviator(stress.value)
    s_star = util.deviator(stress_star)
    s_star_step = util.deviator(stress_star_step)
    del_s_star = s_star - s_star_step

    # stress_3
    q = util.eq(stress.value)
    q_star = util.eqf(stress.value, f)
    p = util.mean(stress.value)
    p_star = util.mean(stress_star)
    p_step = util.mean(stress.value_step)
    p_star_step = util.mean(stress_star_step)

    # strain
    delta_strain_pl = strain_pl.value - strain_pl.value_step
    delepd = util.deviator(delta_strain_pl)
    eps_vol = 3.0 * util.mean(strain.value)
    eps_vol_step = 3.0 * util.mean(strain.value_step)
    gamma_pl_delta = (2.0 / 3.0) * util.eq(delta_strain_pl)

    # additional terms
    sqterm = sq0
    if q != 0:
        sqterm = s / q

    sfqfterm = sq0
    if q_star != 0:
        sfqfterm = s_star / q_star

    e_gammadel_term = (3.0 / 2.0) * sq0
    if gamma_pl_delta != 0:
        e_gammadel_term = delepd / gamma_pl_delta

    sfqfterm2 = s_star / q_star ** 2.0

    del_s_star_sfqf2_term = util.dot(del_s_star, sfqfterm2)
    e_gammadel_sfqf_term = util.dot(e_gammadel_term, sfqfterm)
    e_gammadel_sq_term = util.dot(e_gammadel_term, sqterm)
    if gamma_pl_delta == 0:
        e_gammadel_sfqf_term = 1.0
        e_gammadel_sq_term = 1.0

    # ===  residual calculation
    residual = np.zeros((33, 1))

    residual[0][0] = yield_shear_fabric(strain_pl, smp, vr, f, stress)
    residual[1][0] = yield_cap(stress, strain_pl_c, smp, f)
    residual[26][0] = vr.value - (1.0 + vr.value_step) * np.exp(eps_vol_step - eps_vol) + 1.0

    if not shear.active:
        residual[0][0] = 0.0

    if not cap.active:
        residual[1][0] = 0.0

    for i in range(6):
        residual[2 + i][0] = strain_pl_c.value[i][0] - strain_pl_c.value_step[i][0] - delta_lambda_c.value * \
                             delta[i][0] / 3.0

        residual[8 + i][0] = strain_pl.value[i][0] - strain_pl.value_step[i][0] - strain_pl_c.value[i][0] \
                             + strain_pl_c.value_step[i][0] - delta_lambda_s.value * \
                             ((-1.0 / 3.0) * e_gammadel_sq_term * sinpsim_f * delta[i][0]
                              + (3.0 / 2.0) * sqterm[i][0])

        residual[14 + i][0] = stress.value[i][0] - stress.value_step[i][0] - del_stress_total[i][0] \
                              + del_stress_pl[i][0]

        residual[27 + i][0] = f.value[i][0] - f.value_step[i][0] \
                              - smp.chi * (p * (stress.value[i][0] - stress.value_step[i][0])
                                           - (p - p_step) * stress.value_step[i][0]) / p ** 2.0

    for i in key_stress:
        residual[20 + i][0] = stress.value[i][0] - stress.value_in[i][0] - load.incre_stress[i][0] * (load.istep + 1)

    for i in key_strain:
        residual[20 + i][0] = strain.value[i][0] - strain.value_in[i][0] - load.incre_strain[i][0] * (load.istep + 1)

    return residual, key_stress, key_strain


def get_derivatives(stress, smp, vr, strain_pl_c, delta_lambda_s, ksec, shear, cap, key_stress, key_strain,
                    strain, f, strain_pl):
    # ===  initial calculations:

    # constants:
    mm_f, _, _, sinpsim_f, _, b, a_f = yield_params(smp, vr, stress, strain_pl, f)

    guide = np.array([[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]])

    delta_ten = np.zeros((3, 3))
    ident = np.zeros((6, 6))
    delta = np.zeros((6, 1))
    deldel = np.zeros((6, 6))
    sq0 = np.zeros((6, 1))
    for i in range(6):
        sq0[i][0] = np.sqrt(2.0 / 3.0)
        if i < 3:
            delta[i][0] = 1
            delta_ten[i][i] = 1.0
            sq0[i][0] = 2.0 / 3.0

        for j in range(6):
            if i < 3 and j < 3:
                deldel[i][j] = 1

            if i == j:
                ident[i][j] = 1

    # stress_1
    stress_star_inv = util.invstarinvf(stress.value, f)
    stress_star_inv_ten = util.sixtothree(stress_star_inv)
    stress_star_inv_stress_n = util.invstarinvfn(stress, f)
    stress_star_inv_stress_n_ten = util.sixtothree(stress_star_inv_stress_n)
    stress_star = util.starinvf(stress.value, f)
    stress_star_step = util.starinvf(stress.value_step, f)
    del_stress = stress.value - stress.value_step
    del_stress_abs = np.abs(del_stress)
    del_stress_f = util.dot(f.value, del_stress)
    del_stress_abs_f = util.dot(f.value, del_stress_abs)
    del_stress2 = util.dot(del_stress, del_stress)
    stress_ten = util.sixtothree(stress.value)
    stress_step_ten = util.sixtothree(stress.value_step)

    # stress_2
    s_star = util.deviator(stress_star)
    s_star_step = util.deviator(stress_star_step)
    s = util.deviator(stress.value)
    del_s_star = s_star - s_star_step

    # stress_3
    q = util.eq(stress.value)
    p = util.mean(stress.value)
    p_step = util.mean(stress.value_step)
    q_star = util.eqf(stress.value, f)
    p_star = util.mean(stress_star)
    p_star_step = util.mean(stress_star_step)

    # stress_4
    j2_val = util.j2(stress.value)
    j3_val = util.j3(stress.value)
    s2 = util.ss(stress.value)

    # strain
    delta_strain_pl = strain_pl.value - strain_pl.value_step
    strain_pl_star = util.starf(strain_pl.value, f)
    e_pl_star = util.deviator(strain_pl_star)
    gamma_pl = (2.0 / 3.0) * util.eq(strain_pl.value)
    gamma_pl_star = (2.0 / 3.0) * util.eq(strain_pl_star)
    gamma_pl_delta = (2.0 / 3.0) * util.eq(delta_strain_pl)
    eps_vol = 3.0 * util.mean(strain.value)
    eps_vol_step = 3.0 * util.mean(strain.value_step)
    delepd = util.deviator(delta_strain_pl)
    strain_pl_c_mean = util.mean(strain_pl_c.value)
    if np.abs(strain_pl_c_mean) < 1e-10:
        strain_pl_c_mean = 0.0

    # fabric
    f2 = util.dot(f.value, f.value)
    f_ten = util.sixtothree(f.value)
    finv_ten = np.linalg.inv(f_ten)
    finv = util.threetosix(finv_ten)

    # additional terms_1
    e_gammadel_term = (3.0 / 2.0) * sq0
    if gamma_pl_delta != 0:
        e_gammadel_term = delepd / gamma_pl_delta

    sqterm = sq0
    if q != 0:
        sqterm = s / q

    sfqfterm = sq0
    if q_star != 0:
        sfqfterm = s_star / q_star

    sfqfterm2 = s_star / q_star ** 2.0

    e_gammadel_sfqf_term = util.dot(e_gammadel_term, sfqfterm)
    e_gammadel_sq_term = util.dot(e_gammadel_term, sqterm)
    if gamma_pl_delta == 0:
        e_gammadel_sfqf_term = 1.0
        e_gammadel_sq_term = 1.0

    if gamma_pl_star == 0:
        e_gamma_term = sq0
    else:
        e_gamma_term = (2.0 / 3.0) * e_pl_star / gamma_pl_star

    if q == 0:
        lmbdgammaterm = 1.0
    else:
        lmbdgammaterm = 0.0

    if gamma_pl_delta != 0:
        lmbdgammaterm = delta_lambda_s.value / gamma_pl_delta

    sign_stress = np.zeros((6, 1))
    for i in range(6):
        sign_stress[i][0] = np.sign(del_stress[i][0])
        if del_stress[i][0] == 0.0:
            sign_stress[i][0] = 1.0

    # additional term_2
    del_s_sfqf2_term = util.dot(del_s_star, sfqfterm2)
    del_s_sfqf_term = util.dot(del_s_star, sfqfterm)

    e_gammadel_ten = util.sixtothree(e_gammadel_term)
    e_gammadel_finv = np.matmul(e_gammadel_ten, finv_ten)
    finv_e_gammadel = np.matmul(finv_ten, e_gammadel_ten)
    e_gammadel_finv_vec = util.threetosix(e_gammadel_finv)
    finv_e_gammadel_vec = util.threetosix(finv_e_gammadel)

    s_star_ten = util.sixtothree(s_star)
    s_star_finv = np.matmul(s_star_ten, finv_ten)
    finv_s_star = np.matmul(finv_ten, s_star_ten)
    s_star_finv_vec = util.threetosix(s_star_finv)
    finv_s_star_vec = util.threetosix(finv_s_star)

    e_gamma_term_ten = util.sixtothree(e_gamma_term)
    e_gamma_f = np.matmul(e_gamma_term_ten, f_ten)
    f_e_gamma = np.matmul(f_ten, e_gamma_term_ten)
    e_gamma_f_vec = util.threetosix(e_gamma_f)
    f_e_gamma_vec = util.threetosix(f_e_gamma)

    strain_pl_ten = util.sixtothree(strain_pl.value)
    e_gamma_strain_pl = np.matmul(e_gamma_term_ten, strain_pl_ten)
    strain_pl_e_gamma = np.matmul(strain_pl_ten, e_gamma_term_ten)
    e_gamma_strain_pl_vec = util.threetosix(e_gamma_strain_pl)
    strain_pl_e_gamma_vec = util.threetosix(strain_pl_e_gamma)

    stress_star_inv_stress_step = np.matmul(stress_star_inv_ten, stress_step_ten)
    stress_star_inv_stress_step_finv = np.matmul(stress_star_inv_stress_step, finv_ten)
    stress_step_finv = np.matmul(stress_step_ten, finv_ten)

    stress_step_finv_finv = np.matmul(stress_step_finv, finv_ten)
    finv_stress_step_finv = np.matmul(finv_ten, stress_step_finv)

    # additional terms_3
    finv_i_finv = np.zeros((6, 6))
    stress_starinv_i_stress_starinv = np.zeros((6, 6))
    stress_f_inv = np.matmul(stress_ten, finv_ten)
    aaa = np.zeros((6, 6))
    delta4_finv = np.zeros((6, 6))
    bbb = np.zeros((6, 6))
    eee = np.zeros((6, 6))
    jjj = np.zeros((6, 6))
    kkk = np.zeros((6, 6))
    nnn = np.zeros((6, 6))
    ooo = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ii = guide[i][0]
            jj = guide[i][1]
            kk = guide[j][0]
            ll = guide[j][1]

            finv_i_finv[i][j] = (finv_ten[kk][ii] * finv_ten[jj][ll] + finv_ten[kk][jj] * finv_ten[ii][ll]) / 2.0

            stress_starinv_i_stress_starinv[i][j] = (stress_star_inv_ten[ii][kk] * stress_star_inv_ten[ll][jj]
                                                     + stress_star_inv_ten[ii][ll] * stress_star_inv_ten[kk][jj]) / 2.0

            aaa[i][j] = (stress_star_inv_ten[ii][kk] * stress_star_inv_stress_n_ten[ll][jj]
                         + stress_star_inv_stress_n_ten[ii][ll] * stress_star_inv_ten[kk][jj]) / 2.0

            delta4_finv[i][j] = (delta_ten[ii][kk] * finv_ten[ll][jj] + delta_ten[ii][ll] * finv_ten[kk][jj]) / 2.0

            bbb[i][j] = (- aaa[i][j] + (p_star_step - p_star) * stress_starinv_i_stress_starinv[i][j]
                         + stress_star_inv[i][0] * delta[j][0] / 3.0 - (3.0 / 2.0) * del_s_sfqf2_term *
                         (delta[j][0] * stress_star_inv[i][0] / 3.0 - p_star * stress_starinv_i_stress_starinv[i][j])
                         + (3.0 / 2.0) * (delta[i][0] - p_star * stress_star_inv[i][0]) *
                         (sfqfterm2[j][0] + (q_star * del_s_star[j][0] - 3.0 * s_star[j][0] * del_s_sfqf_term) /
                          q_star ** 3.0))

            eee[i][j] = (stress_f_inv[ii][kk] * finv_ten[ll][jj] + stress_f_inv[ii][ll] * finv_ten[kk][jj]) / 2.0

            jjj[i][j] = (stress_star_inv_stress_step_finv[ii][kk] * finv_ten[ll][jj]
                         + stress_star_inv_stress_step_finv[ii][ll] * finv_ten[kk][jj]) / 2.0

            kkk[i][j] = (1.0 / 3.0) * stress_star_inv_ten[ii][jj] * (finv_stress_step_finv[ll][kk]
                                                                     + finv_stress_step_finv[kk][ll]) / 2.0

            nnn[i][j] = (delta_ten[ii][kk] * delta_ten[jj][ll] + delta_ten[ii][ll] * delta_ten[jj][kk]) / 2.0 \
                        - delta_ten[ii][jj] * delta_ten[kk][ll] / 3.0

            ooo[i][j] = (stress_step_finv[ii][kk] * finv_ten[ll][jj]
                         + stress_step_finv[ii][ll] * finv_ten[kk][jj]) / 2.0

            if j > 2:
                finv_i_finv[i][j] = finv_i_finv[i][j] * np.sqrt(2.0)
                stress_starinv_i_stress_starinv[i][j] = stress_starinv_i_stress_starinv[i][j] * np.sqrt(2.0)
                aaa[i][j] = aaa[i][j] * np.sqrt(2.0)
                bbb[i][j] = bbb[i][j] * np.sqrt(2.0)
                eee[i][j] = eee[i][j] * np.sqrt(2.0)
                jjj[i][j] = jjj[i][j] * np.sqrt(2.0)
                kkk[i][j] = kkk[i][j] * np.sqrt(2.0)
                nnn[i][j] = nnn[i][j] * np.sqrt(2.0)
                ooo[i][j] = ooo[i][j] * np.sqrt(2.0)
                delta4_finv[i][j] = delta4_finv[i][j] * np.sqrt(2.0)

        if i > 2:
            for jj in range(6):
                finv_i_finv[i][jj] = finv_i_finv[i][jj] * np.sqrt(2.0)
                stress_starinv_i_stress_starinv[i][jj] = stress_starinv_i_stress_starinv[i][jj] * np.sqrt(2.0)
                aaa[i][jj] = aaa[i][jj] * np.sqrt(2.0)
                bbb[i][jj] = bbb[i][jj] * np.sqrt(2.0)
                eee[i][jj] = eee[i][jj] * np.sqrt(2.0)
                jjj[i][jj] = jjj[i][jj] * np.sqrt(2.0)
                kkk[i][jj] = kkk[i][jj] * np.sqrt(2.0)
                nnn[i][jj] = nnn[i][jj] * np.sqrt(2.0)
                ooo[i][jj] = ooo[i][jj] * np.sqrt(2.0)
                delta4_finv[i][jj] = delta4_finv[i][jj] * np.sqrt(2.0)

    ccc = np.matmul(bbb, delta4_finv)
    ddd = np.matmul(np.transpose(finv_i_finv), stress.value)
    ggg = np.matmul(np.transpose(eee), e_gammadel_term)
    hhh = np.matmul(np.transpose(eee), s_star)
    iii = - np.matmul(bbb, eee)
    mmm = np.matmul(nnn, ooo)
    lll = np.matmul(np.transpose(mmm), sfqfterm2)

    # mathematica terms
    const1 = ((np.exp((p_star / smp.hcr) ** smp.ncr) * vr.value) / smp.vrc0) ** smp.beta
    if math.isnan(const1):
        print("const1 is NaN")
        sys.exit()

    if q == 0:
        const2 = 1.0
    else:
        const2 = (3.0 * np.sqrt(3.0) / 2.0) * (1.0 / j2_val) ** (3.0 / 2.0) * j3_val

    const3 = ((np.exp((p_star / smp.hcr) ** smp.ncr) * vr.value) / smp.vrc0) ** smp.alpha
    if math.isnan(const3):
        print("const3 is NaN")
        sys.exit()

    dmmfdpf = -(36.0 * smp.ncr * (p_star / smp.hcr) ** smp.ncr * smp.sinphicv * const1 * smp.beta * gamma_pl_star
                    * (a_f + gamma_pl_star) * smp.mu) / (p_star * (smp.sinphicv * gamma_pl_star - 3.0 * const1 *
                                                                   (a_f + gamma_pl_star)) ** 2.0 *
                                                         (1.0 + smp.mu + (-1.0 + smp.mu) * const2))
    if math.isnan(dmmfdpf):
        print("dmmfdpf is NaN")
        sys.exit()

    dmmfdgammapf = (36.0 * a_f * smp.sinphicv * const1 * smp.mu) / ((smp.sinphicv * gamma_pl_star -
                                                                        3.0 * const1 * (a_f + gamma_pl_star)) ** 2.0 *
                                                                        (1.0 + smp.mu + (-1.0 + smp.mu) * const2))

    dmmfdaf = -(36.0 * gamma_pl_star * smp.sinphicv * const1 * smp.mu) / ((smp.sinphicv * gamma_pl_star -
                                                                        3.0 * const1 * (a_f + gamma_pl_star)) ** 2.0 *
                                                                        (1.0 + smp.mu + (-1.0 + smp.mu) * const2))

    if q == 0:
        dmmfdj3 = 1.0
    else:
        dmmfdj3 = (72.0 * np.sqrt(3.0) * smp.sinphicv * gamma_pl_star * (-1.0 + smp.mu) * smp.mu) / \
                (4.0 * np.sqrt(j2_val ** 3.0) * (smp.sinphicv * gamma_pl_star - 3.0 * const1 * (a_f + gamma_pl_star)) *
                (const2 * (-1.0 + smp.mu) + (1.0 + smp.mu)) ** 2.0)

    if q == 0:
        dmmfdj2 = 1.0
    else:
        dmmfdj2 = -(18.0 * const2 * smp.sinphicv * gamma_pl_star * (-1.0 + smp.mu) * smp.mu) / \
            (j2_val * (smp.sinphicv * gamma_pl_star - 3.0 * const1 * (a_f + gamma_pl_star)) *
                (const2 * (-1.0 + smp.mu) + (1.0 + smp.mu)) ** 2.0)

    dmmfdvr = -(36.0 * smp.sinphicv * const1 * smp.beta * gamma_pl_star * (a_f + gamma_pl_star) * smp.mu) / \
        (vr.value * (smp.sinphicv * gamma_pl_star - 3.0 * const1 * (a_f + gamma_pl_star)) ** 2.0 *
            (const2 * (-1.0 + smp.mu) + (1.0 + smp.mu)))

    dsinpsimfdpf = (smp.ncr * (p_star / smp.hcr) ** smp.ncr * smp.sinphicv *
                        (-const3 * smp.alpha * (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star) +
                         (smp.sinphicv ** 2.0 * (const3 / const1 ** 2.0) * smp.alpha * gamma_pl_star ** 2.0 *
                          (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star)) / (a_f + gamma_pl_star) ** 2.0 +
                         (1.0 / const1) * smp.beta * gamma_pl_star * (-(smp.b_0 + gamma_pl_star) ** 2.0 +
                                                                      smp.sinphicv ** 2.0 * const3 ** 2.0 *
                                                                      (b + gamma_pl_star) ** 2.0) /
                         (a_f + gamma_pl_star))) / (p_star * (smp.b_0 + gamma_pl_star - (smp.sinphicv ** 2.0 *
                                                                                         (const3 / const1) *
                                                                                         gamma_pl_star *
                                                                                         (b + gamma_pl_star)) /
                                                              (a_f + gamma_pl_star)) ** 2.0)
    if math.isnan(dsinpsimfdpf):
        print("dsinpsimfdpf is NaN")
        sys.exit()

    dsinpsimfdgammapf = (a_f * smp.sinphicv * const1 * (smp.b_0 + gamma_pl_star) ** 2.0 -
                         a_f * smp.sinphicv ** 3.0 * const1 * const3 ** 2.0 * (b + gamma_pl_star) ** 2.0 +
                         (smp.b_0 - b) * smp.sinphicv * const3 * (smp.sinphicv ** 2.0 * gamma_pl_star ** 2.0 -
                                                                        const1 ** 2.0 * (a_f +
                                                                                         gamma_pl_star) ** 2.0)) / \
                        (const1 ** 2.0 * (a_f + gamma_pl_star) ** 2.0 * (smp.b_0 + gamma_pl_star) ** 2.0 - 2.0 *
                         smp.sinphicv ** 2.0 * const1 * const3 * gamma_pl_star * (a_f + gamma_pl_star) *
                         (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star) + smp.sinphicv ** 4.0 * const3 ** 2.0 *
                         gamma_pl_star ** 2.0 * (b + gamma_pl_star) ** 2.0)

    dsinpsimfdbf = (smp.sinphicv * const3 * (smp.b_0 + gamma_pl_star) * (smp.sinphicv ** 2.0 * gamma_pl_star ** 2.0 -
                                                                      const1 ** 2.0 * (a_f +
                                                                                       gamma_pl_star) ** 2.0)) / \
                   (const1 ** 2.0 * (a_f + gamma_pl_star) ** 2.0 * (smp.b_0 + gamma_pl_star) ** 2.0 - 2.0 *
                    smp.sinphicv ** 2.0 * const1 * const3 * gamma_pl_star * (a_f + gamma_pl_star) *
                    (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star) + smp.sinphicv ** 4.0 * const3 ** 2.0 *
                    gamma_pl_star ** 2.0 * (b + gamma_pl_star) ** 2.0)

    dsinpsimfdvr = (smp.sinphicv * (- const3 * smp.alpha * (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star) +
                                    (smp.sinphicv ** 2.0 * (const3 / const1 ** 2.0) * smp.alpha * gamma_pl_star ** 2.0 *
                                     (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star)) / (a_f +
                                                                                         gamma_pl_star) ** 2.0 +
                                    ((1.0 / const1) * smp.beta * gamma_pl_star * (-(smp.b_0 + gamma_pl_star) ** 2.0 +
                                                                               smp.sinphicv ** 2.0 * const3 ** 2.0 *
                                                                               (b + gamma_pl_star) ** 2.0)) /
                                    (a_f + gamma_pl_star))) / (vr.value * (smp.b_0 + gamma_pl_star -
                                                                            (smp.sinphicv ** 2.0 *
                                                                             (const3 / const1) * gamma_pl_star *
                                                                             (b + gamma_pl_star)) /
                                                                            (a_f + gamma_pl_star)) ** 2.0)

    dsinpsimfdaf = (smp.sinphicv * const1 * gamma_pl_star * (-(smp.b_0 + gamma_pl_star) ** 2.0 + smp.sinphicv ** 2.0 *
                                                          const3 ** 2.0 * (b + gamma_pl_star) ** 2.0)) / \
                   (const1 ** 2 * (a_f + gamma_pl_star) ** 2.0 * (smp.b_0 + gamma_pl_star) ** 2.0 -
                    2.0 * smp.sinphicv ** 2 * const1 * const3 * gamma_pl_star * (a_f + gamma_pl_star) *
                    (smp.b_0 + gamma_pl_star) * (b + gamma_pl_star) + smp.sinphicv ** 4.0 * const3 ** 2.0 *
                    gamma_pl_star ** 2.0 * (b + gamma_pl_star) ** 2.0)

    # ===  derivative calculations
    derivative = np.zeros((33, 33))

    derivative[0][26] = - p * dmmfdvr
    derivative[26][26] = 1.0

    for j in range(6):
        derivative[0][j + 2] = (3.0 / 2.0) * sqterm[j][0] \
                               - mm_f / 3.0 * delta[j][0] \
                               - p * (dmmfdpf / 3.0 * finv[j][0]
                                      + dmmfdj2 * s[j][0]
                                      + dmmfdj3 * (s2[j][0] - (2.0 / 3.0) * j2_val * delta[j][0]))

        derivative[0][j + 8] = - p * dmmfdgammapf * (e_gamma_f_vec[j][0] + f_e_gamma_vec[j][0]) / 2.0

        derivative[0][j + 27] = - p * (-dmmfdpf * ddd[j][0] / 3.0
                                       + dmmfdgammapf * (e_gamma_strain_pl_vec[j][0] + strain_pl_e_gamma_vec[j][0]) /
                                       2.0)

        derivative[1][j + 27] = - (1.0 / 3.0) * ddd[j][0]

        derivative[1][j + 2] = 1.0 / 3.0 * delta[j][0]

        derivative[1][j + 14] = - delta[j][0] * smp.p0 * (strain_pl_c_mean * 3.0 / smp.hic) ** \
            ((1.0 - smp.nic) / smp.nic) / (smp.nic * smp.hic)

        derivative[26][j + 20] = delta[j][0] * (1.0 + vr.value_step) * np.exp(eps_vol_step - eps_vol)

    for i in range(6):
        derivative[i + 2][1] = - 1.0 / 3.0 * delta[i][0]

        derivative[i + 8][0] = - (3.0 / 2.0) * sqterm[i][0] \
                               + e_gammadel_sq_term * sinpsim_f * delta[i][0] / 3.0

        derivative[i + 8][26] = e_gammadel_sq_term * delta_lambda_s.value * delta[i][0] * dsinpsimfdvr / 3.0

        for j in range(6):
            derivative[i + 2][j + 14] = ident[i][j]

            if q != 0:
                derivative[i + 8][j + 2] = - (3.0 / 2.0) * delta_lambda_s.value * (1 / q ** 2.0) * \
                                           (q * (ident[i][j] - deldel[i][j] / 3.0)
                                            - (3.0 / 2.0) * sqterm[i][0] * s[j][0]) \
                                           + e_gammadel_sq_term * (1.0 / 3.0) * delta[i][0] * delta_lambda_s.value * \
                                           ((1.0 / 3.0) * dsinpsimfdpf * finv[j][0]) \
                                           + (1.0 / 3.0) * delta_lambda_s.value * sinpsim_f * delta[i][0] * \
                                           (q * e_gammadel_term[j][0]
                                            - (3.0 / 2.0) * e_gammadel_sq_term * s[j][0]) / q ** 2.0

            derivative[i + 8][j + 8] = ident[i][j] \
                                       + e_gammadel_sq_term * delta[i][0] * (1.0 / 3.0) * delta_lambda_s.value * \
                                       dsinpsimfdgammapf * (e_gamma_f_vec[j][0] + f_e_gamma_vec[j][0]) / 2.0 \
                                       + (1.0 / 3.0) * lmbdgammaterm * sinpsim_f * delta[i][0] \
                                       * (sqterm[j][0] - (2.0 / 3.0) * e_gammadel_term[j][0] * e_gammadel_sq_term)

            derivative[i + 8][j + 14] = - ident[i][j]

            derivative[i + 8][j + 27] = (1.0 / 3.0) * e_gammadel_sq_term * delta_lambda_s.value * delta[i][0] \
                                        * (dsinpsimfdgammapf * (e_gamma_strain_pl_vec[j][0]
                                                                + strain_pl_e_gamma_vec[j][0]) / 2.0
                                           - (1.0 / 3.0) * dsinpsimfdpf * ddd[j][0]) \
                                        * lll[j][0]

            derivative[i + 14][j + 2] = ident[i][j]
            derivative[i + 14][j + 8] = ksec * smp.cebar[i][j]
            derivative[i + 14][j + 20] = - ksec * smp.cebar[i][j]

            derivative[i + 27][j + 2] = (2.0 / 3.0) * delta[j][0] * (smp.chi / p ** 3.0) * \
                                        (p * (stress.value[i][0] - stress.value_step[i][0])
                                         - stress.value_step[i][0] * (p - p_step)) \
                                        - (smp.chi / p ** 2.0) * (p * ident[i][j] + delta[j][0] *
                                                                  (stress.value[i][0] - stress.value_step[i][0]) / 3.0
                                                                  - stress.value_step[i][0] * delta[j][0] / 3.0)

            derivative[i + 27][j + 27] = ident[i][j]

            if j in key_stress:
                derivative[j + 20][i + 2] = ident[j][i]

            if j in key_strain:
                derivative[j + 20][i + 20] = ident[j][i]

    if not shear.active:
        derivative[0, 0:33] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0]
        derivative[0:33, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0]
        derivative[0][0] = 1.0

    if not cap.active:
        derivative[1, 0:33] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0]
        derivative[0:33, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0]
        derivative[1][1] = 1.0

    return derivative


def plastic_iteration(func_arg, criteria, iteration):

    smp, ksec, istep, load, shear, cap = func_arg
    stress, strain_pl, strain_pl_c, delta_lambda_s, delta_lambda_c, strain, vr, f = criteria

    residual, key_stress, key_strain = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c, delta_lambda_s,
                                                     delta_lambda_c, load, strain, istep, shear, cap, f)
    # calculate derivatives analytically
    try:
        derivative = get_derivatives(stress, smp, vr, strain_pl_c, delta_lambda_s, ksec, shear, cap, key_stress,
                                     key_strain, strain, f, strain_pl)
    except ZeroDivisionError:
        print("Division by zero in calculating the derivatives of the return mapping")
        sys.exit()

    # calculate derivatives numerically
    # derivative_num = numerical_derivative(delta_lambda_s, delta_lambda_c, vr, f, stress, strain, strain_pl,
    #                                       strain_pl_c, ksec, smp, shear, cap, load, residual, istep)
    # derivative = derivative_num

    # debug (numerical calculation of derivatives):
    # if istep == 0 and iteration == 2:
    #
    #     derivative_num = numerical_derivative(delta_lambda_s, delta_lambda_c, vr, f, stress, strain, strain_pl,
    #                                           strain_pl_c, ksec, smp, shear, cap, load, residual, istep)
    #
    #     kashk = 1
    # debug

    # determinant = np.linalg.det(derivative)
    # condition = np.linalg.cond(derivative)

    try:
        dstate = - np.matmul(np.linalg.inv(derivative), residual)
    except np.linalg.LinAlgError:
        print("Derivative matrix is singular in return mapping: step " + str(istep + 1))
        sys.exit()

    delta_lambda_s.update(dstate[0][0])
    delta_lambda_c.update(dstate[1][0])
    stress.update(dstate[2:8, 0:1])
    strain_pl.update(dstate[8:14, 0:1])
    strain_pl_c.update(dstate[14:20, 0:1])
    strain.update(dstate[20:26, 0:1])
    vr.update(dstate[26][0])
    f.update(dstate[27:33, 0:1])

    error = np.sqrt(delta_lambda_s.error() ** 2.0 + delta_lambda_c.error() ** 2.0 + stress.error() ** 2.0 +
                    strain_pl.error() ** 2.0 + strain_pl_c.error() ** 2.0 + strain.error() ** 2.0 +
                    vr.error() ** 2.0 + f.error() ** 2.0)

    # debug
    # print('return mapping residual: ' + str(error))

    delta_lambda_s.set_iter()
    delta_lambda_c.set_iter()
    stress.set_iter()
    strain_pl.set_iter()
    strain_pl_c.set_iter()
    strain.set_iter()
    vr.set_iter()
    f.set_iter()

    return [error, [stress, strain_pl, strain_pl_c, delta_lambda_s, delta_lambda_c, strain, vr, f]]


def return_map(slvr, stress, strain, strain_pl, strain_pl_c, smp, vr, ksec, istep, load, mystep, f, shear, cap):

    delta_lambda_s = StateScalar(initial_value=0.0)
    delta_lambda_c = StateScalar(initial_value=0.0)

    flagplas2 = 0
    iiter = 0
    while flagplas2 != 0 or iiter == 0:
        iiter += 1
        delta_lambda_s.reset()
        delta_lambda_c.reset()
        stress.reset()
        strain.reset()
        strain_pl.reset()
        strain_pl_c.reset()
        vr.reset()
        f.reset()

        mix = [stress, strain_pl, strain_pl_c, delta_lambda_s, delta_lambda_c, strain, vr, f]
        plastic_out, conflag, iteration, error_flag = iterator(func=plastic_iteration,
                                                               func_arg=[smp, ksec, istep, load, shear, cap],
                                                               toler=slvr.tol_rtmp, error_in=slvr.err_in,
                                                               criteria=mix, maxiter=slvr.maxit_rtmp)

        if conflag == 1:
            print("Maxiumum iterations reached in return mapping at step " + str(istep))
            pickle.dump([mystep.stress, mystep.strain, mystep.strain_pl, mystep.strain_pl_c, mystep.vr, mystep.p,
                         mystep.q, mystep.gamma_pl, mystep.gamma, mystep.axial_strain, mystep.axial_stress, mystep.f,
                         mystep.sphm, mystep.mmf, mystep.gamma_pl_star],
                        open('output.dat', "wb"))
            sys.exit()
        elif error_flag == 1:
            print('error is increasing in 5 consecutive iterations in return mapping')
            pickle.dump([mystep.stress, mystep.strain, mystep.strain_pl, mystep.strain_pl_c, mystep.vr, mystep.p,
                         mystep.q, mystep.gamma_pl, mystep.gamma, mystep.axial_strain, mystep.axial_stress, mystep.f,
                         mystep.sphm, mystep.mmf, mystep.gamma_pl_star],
                        open('output.dat', "wb"))
            sys.exit()
        else:
            if slvr.stpinfo:
                print("return mapping converged with " + str(iteration) + " iterations")

        stress, strain_pl, strain_pl_c, delta_lambda_s, delta_lambda_c, strain, vr, f = plastic_out[1]

        flagplas = 0
        if delta_lambda_s.value < 0:
            # for debug
            print('shear part is being deactivated ...')
            # input('shear part is being deactivated. Press enter to continue ...')
            flagplas = 1
            shear.deactivate()
            # sys.exit()

        if delta_lambda_c.value < 0:
            # for debug
            print('cap part is being deactivated ...')
            # input('cap part is being deactivated. Press enter to continue ...')
            flagplas = 1
            cap.deactivate()

    stress.set_step()
    strain.set_step()
    strain_pl.set_step()
    strain_pl_c.set_step()
    vr.set_step()
    f.set_step()

    return stress, strain_pl, strain_pl_c, strain, vr, f, flagplas


def set_in_state(vr_in, strsin, fin, strnin, strnpin, strnpcin):
    void_ratio = StateScalar(initial_value=vr_in)

    stress = StateVector(initial_value=strsin)
    fabric = StateVector(initial_value=fin)

    strain_pl = StateVector(initial_value=strnpin)  # total plastic strain
    strain_pl_cap = StateVector(initial_value=strnpcin)  # cap plastic strain
    strain = StateVector(initial_value=strnin)  # total strain

    return void_ratio, fabric, stress, strain, strain_pl, strain_pl_cap


def print_step(slvr, istep):
    if slvr.stpinfo or (istep + 1) % slvr.freq == 0:
        print("====STEP " + str(istep + 1))


def yield_params(smp, vr, stress, strain_pl, f):
    stress_star = util.starinvf(stress.value, f)
    p_f = util.mean(stress_star)
    strain_pl_star = util.starf(strain_pl.value, f)
    strain_pl_j2_star = util.j2(strain_pl_star)
    gamma_pl_star = np.sqrt((4.0 / 3.0) * strain_pl_j2_star)
    lode_val = util.lode(stress.value)
    vrc_f = smp.vrc0 * np.exp(- (p_f / smp.hcr) ** smp.ncr)
    if math.isnan(vrc_f):
        print("vrc_f in yield_params is NaN. Not important.")

    delstress = stress.value - stress.value_step
    a_f = smp.a_f
    sinphim_f = (vr.value / vrc_f) ** (-smp.beta) * (gamma_pl_star / (gamma_pl_star + a_f)) * smp.sinphicv
    b = smp.x
    sinphif = (vr.value / vrc_f) ** smp.alpha * ((gamma_pl_star + b) / (gamma_pl_star + smp.b_0)) * smp.sinphicv
    sinpsim_f = (sinphim_f - sinphif) / (1.0 - sinphim_f * sinphif)
    mtc_f = 6.0 * sinphim_f / (3.0 - sinphim_f)
    mm_f = 2.0 * smp.mu * mtc_f / ((1.0 + smp.mu) - (1.0 - smp.mu) * lode_val)
    return mm_f, sinphim_f, mtc_f, sinpsim_f, vrc_f, b, a_f


def reach_goal(load, solver, sample, void_ratio, stress, strain, strain_pl, strain_pl_cap, fabric, mystep, kstep):

    flagplas = 0
    for istepg in range(load.nstep):

        istep = istepg + kstep
        load.istep += 1

        print_step(slvr=solver, istep=istep)

        ksec, stress, strain, void_ratio, \
            fabric, strain_pl, strain_pl_cap = elastic_trial(slvr=solver, smp=sample, vr=void_ratio, load=load,
                                                             stress=stress, istep=istep, strain=strain,
                                                             strain_pl=strain_pl, strain_pl_c=strain_pl_cap, f=fabric)

        ifplastic, shear, cap = check_trial(stress=stress, strain_pl=strain_pl, smp=sample, vr=void_ratio,
                                            strain_pl_c=strain_pl_cap, f=fabric)

        if ifplastic:
            stress, strain_pl, strain_pl_cap, strain, void_ratio, fabric, flagplas \
                = return_map(slvr=solver, stress=stress, strain=strain,  strain_pl=strain_pl,
                             strain_pl_c=strain_pl_cap, smp=sample, vr=void_ratio, ksec=ksec, istep=istep,
                             load=load, mystep=mystep, f=fabric, shear=shear, cap=cap)
        else:
            stress, strain, void_ratio, fabric = accept_trial(stress=stress, strain=strain, vr=void_ratio, f=fabric)

        # _, _, _, _, _, mmff, sinphimf, _, _, _, _, _ = yield_params(sample, void_ratio, stress, strain_pl, fabric)
        mystep.record(strain=strain, stress=stress, strain_pl=strain_pl, strain_pl_c=strain_pl_cap,
                      vr=void_ratio, f=fabric, sphm=1.0, mmf=1.0, ksecant=ksec)

        time.sleep(solver.stp_pause)

    return mystep, void_ratio, fabric, strain, stress, strain_pl, strain_pl_cap, flagplas


def initialize(bdplane, pangle, f_in, conf, smp):

    bdplane = bdplane * np.pi / 180.0
    pangle = pangle * np.pi / 180.0
    f_global = np.zeros((3, 3))
    f_global[0][0] = f_in[1] * np.sin(pangle) ** 2.0 + np.cos(pangle) ** 2.0 * \
                     (f_in[0] * np.cos(bdplane) ** 2.0 + f_in[2] * np.sin(bdplane) ** 2.0)

    f_global[0][1] = np.cos(pangle) * np.sin(pangle) * \
                     (-f_in[1] + f_in[0] * np.cos(bdplane) ** 2.0 + f_in[2] * np.sin(bdplane) ** 2.0)

    f_global[0][2] = (-f_in[0] + f_in[2]) * np.cos(pangle) * np.cos(bdplane) * np.sin(bdplane)

    f_global[1][0] = f_global[0][1]

    f_global[1][1] = f_in[1] * np.cos(pangle) ** 2.0 + np.sin(pangle) ** 2.0 * \
                     (f_in[0] * np.cos(bdplane) ** 2.0 + f_in[2] * np.sin(bdplane) ** 2.0)

    f_global[1][2] = (-f_in[0] + f_in[2]) * np.sin(pangle) * np.cos(bdplane) * np.sin(bdplane)

    f_global[2][0] = f_global[0][2]
    f_global[2][1] = f_global[1][2]

    f_global[2][2] = f_in[0] * np.sin(bdplane) ** 2.0 + f_in[2] * np.cos(bdplane) ** 2.0

    fabric_global_initial = np.array([[f_global[0][0]], [f_global[1][1]], [f_global[2][2]],
                                      [f_global[1][2] * np.sqrt(2.0)], [f_global[0][2] * np.sqrt(2.0)],
                                      [f_global[0][1] * np.sqrt(2.0)]])

    stress_initial = np.array([[conf], [conf], [conf], [0.0], [0.0], [0.0]])

    f_ten = util.sixtothree(fabric_global_initial)
    finv_ten = np.linalg.inv(f_ten)
    p_f = conf * (finv_ten[0][0] + finv_ten[1][1] + finv_ten[2][2]) / 3.0
    epsv_in = (p_f / smp.p0) ** smp.nic * smp.hic

    strain_pl_f_in = np.zeros((6, 1))
    for i in range(3):
        strain_pl_f_in[i][0] = epsv_in / (finv_ten[0][0] + finv_ten[1][1] + finv_ten[2][2])

    strain_pl_f_ten = util.sixtothree(strain_pl_f_in)
    strain_pl_ten = np.matmul(strain_pl_f_ten, finv_ten)
    strain_plastic_initial = util.threetosix(strain_pl_ten)

    strain_plastic_initial = np.zeros((6, 1))

    return stress_initial, fabric_global_initial, strain_plastic_initial


def numerical_derivative(delta_lambda_s, delta_lambda_c, vr, f, stress, strain, strain_pl, strain_pl_c, ksec, smp,
                         shear, cap, load, residual, istep):
    derivative_num = np.zeros((33, 33))

    incre = 1e-10
    delta_lambda_s.value = delta_lambda_s.value_iter + incre
    residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                        delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                        shear, cap, f)
    derivative_num[0:33, 0:1] = (residual2 - residual) / incre
    delta_lambda_s.value = delta_lambda_s.value_iter

    incre = 1e-10
    delta_lambda_c.value = delta_lambda_c.value_iter + incre
    residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                        delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                        shear, cap, f)
    derivative_num[0:33, 1:2] = (residual2 - residual) / incre
    delta_lambda_c.value = delta_lambda_c.value_iter

    incre = 1e-10
    vr.value = vr.value_iter + incre
    residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                        delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                        shear, cap, f)
    derivative_num[0:33, 26:27] = (residual2 - residual) / incre
    vr.value = vr.value_iter

    incre = 1e-10
    answer = np.zeros((33, 6))
    for j in range(6):
        increm = np.zeros((6, 1))
        increm[j][0] = incre
        stress.value = stress.value_iter + increm
        residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                            delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                            shear, cap, f)
        for i in range(33):
            answer[i][j] = (residual2[i][0] - residual[i][0]) / incre

    derivative_num[0:33, 2:8] = answer
    stress.value = stress.value_iter

    incre = 1e-10
    answer = np.zeros((33, 6))
    for j in range(6):
        increm = np.zeros((6, 1))
        increm[j][0] = incre
        strain_pl.value = strain_pl.value_iter + increm
        residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                            delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                            shear, cap, f)
        for i in range(33):
            answer[i][j] = (residual2[i][0] - residual[i][0]) / incre

    derivative_num[0:33, 8:14] = answer
    strain_pl.value = strain_pl.value_iter

    incre = 1e-10
    answer = np.zeros((33, 6))
    for j in range(6):
        increm = np.zeros((6, 1))
        increm[j][0] = incre
        strain_pl_c.value = strain_pl_c.value_iter + increm
        residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                            delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                            shear, cap, f)
        for i in range(33):
            answer[i][j] = (residual2[i][0] - residual[i][0]) / incre

    derivative_num[0:33, 14:20] = answer
    strain_pl_c.value = strain_pl_c.value_iter

    incre = 1e-10
    answer = np.zeros((33, 6))
    for j in range(6):
        increm = np.zeros((6, 1))
        increm[j][0] = incre
        strain.value = strain.value_iter + increm
        residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                            delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                            shear, cap, f)
        for i in range(33):
            answer[i][j] = (residual2[i][0] - residual[i][0]) / incre

    derivative_num[0:33, 20:26] = answer
    strain.value = strain.value_iter

    incre = 1e-10
    answer = np.zeros((33, 6))
    for j in range(6):
        increm = np.zeros((6, 1))
        increm[j][0] = incre
        f.value = f.value_iter + increm
        residual2, key_stress2, key_strain2 = get_residuals(stress, strain_pl, smp, vr, ksec, strain_pl_c,
                                                            delta_lambda_s, delta_lambda_c, load, strain, istep,
                                                            shear, cap, f)
        for i in range(33):
            answer[i][j] = (residual2[i][0] - residual[i][0]) / incre

    derivative_num[0:33, 27:33] = answer
    f.value = f.value_iter

    if not cap.active:
        derivative_num[1, 0:33] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0]
        derivative_num[0:33, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0]
        derivative_num[1][1] = 1.0

    return derivative_num


def yield_shear_fabric(strain_pl, smp, vr, f, stress):
    p = util.mean(stress.value)
    q = util.eq(stress.value)
    strain_pl_star = util.starf(strain_pl.value, f)
    gamma_pl_star = (2.0 / 3.0) * util.eq(strain_pl_star)
    lode_val = util.lode(stress.value)
    stress_star = util.starinvf(stress.value, f)
    p_star = util.mean(stress_star)
    vrc_f = smp.vrc0 * np.exp(- (p_star / smp.hcr) ** smp.ncr)
    if math.isnan(vrc_f):
        print("vrc_f in yield_shear_fabric is NaN")
        sys.exit()
    delta_stress = stress.value - stress.value_step
    a_f = smp.a_f
    sinphim_f = (vr.value / vrc_f) ** (-smp.beta) * (gamma_pl_star / (gamma_pl_star + a_f)) * smp.sinphicv
    mtc_f = 6.0 * sinphim_f / (3.0 - sinphim_f)
    mm_f = 2.0 * smp.mu * mtc_f / ((1.0 + smp.mu) - (1.0 - smp.mu) * lode_val)
    out = q - mm_f * p
    return out


def yield_cap(stress, strain_pl_c, smp, f):
    strain_pl_c_mean = util.mean(strain_pl_c.value)
    if np.abs(strain_pl_c_mean) < 1e-10:
        strain_pl_c_mean = 0.0

    pc = smp.p0 * (strain_pl_c_mean * 3.0 / smp.hic) ** (1.0 / smp.nic)
    p = util.mean(stress.value)
    out = p - pc
    return out


def random_strain_increment(strain_mag_max, strain_incre_number):
    # strain_mag = 2.0 * np.random.rand(strain_incre_number, 1) * strain_mag_max - strain_mag_max
    strain_mag = np.random.rand(strain_incre_number, 1) * strain_mag_max

    uniform1 = np.random.rand(strain_incre_number, 1)
    uniform2 = np.random.rand(strain_incre_number, 1)
    theta1 = np.zeros((strain_incre_number, 1))
    theta2 = np.zeros((strain_incre_number, 1))
    for i in range(strain_incre_number):
        theta1[i, 0] = math.acos(1.0 - 2.0 * uniform1[i, 0])
        theta2[i, 0] = math.acos(1.0 - 2.0 * uniform2[i, 0])

    phi1 = np.random.rand(strain_incre_number, 1) * 2.0 * math.pi
    phi2 = np.random.rand(strain_incre_number, 1) * 2.0 * math.pi

    strain_incre_all = np.zeros((strain_incre_number, 6))
    strain_incre_mean = np.zeros((strain_incre_number, 1))
    strain_incre_gamma = np.zeros((strain_incre_number, 1))
    for ik in range(strain_incre_number):
        strain1 = np.zeros((6, 1))
        thetachoose = theta1[0, 0]
        phichoose = phi1[0, 0]
        strain1[0][0] = strain_mag[ik, 0] * math.sin(thetachoose) * math.cos(phichoose)
        strain1[1][0] = strain_mag[ik, 0] * math.sin(thetachoose) * math.sin(phichoose)
        strain1[2][0] = strain_mag[ik, 0] * math.cos(thetachoose)
        strain2 = util.sixtothree(strain1)

        trans_1 = np.array([[math.cos(phi2[ik, 0]), math.sin(phi2[ik, 0]), 0.0],
                            [-math.sin(phi2[ik, 0]), math.cos(phi2[ik, 0]), 0.0],
                            [0.0, 0.0, 1.0]])
        trans_2 = np.array([[math.cos(theta2[ik, 0]), 0.0, -math.sin(theta2[ik, 0])],
                            [0.0, 1.0, 0.0],
                            [math.sin(theta2[ik, 0]), 0.0, math.cos(theta2[ik, 0])]])
        trans_3d_inv = np.dot(trans_2, trans_1)
        trans_3d = np.linalg.inv(trans_3d_inv)

        strain3 = np.dot(strain2, np.transpose(trans_3d))
        strain4 = np.dot(trans_3d, strain3)

        # strainf = util.threetosix(strain4)
        # work only in principal space:
        strainf = strain1

        strain_incre_all[ik, :] = np.transpose(strainf)

        strain_incre_mean[ik, 0] = util.mean(strainf)

        strain_incre_j2 = util.j2(strainf)
        strain_incre_gamma[ik, 0] = np.sqrt((4.0 / 3.0) * strain_incre_j2)

    # plt.hist(strain_mag, bins=100)
    # plt.show()
    #
    # plt.hist(strain_incre_mean, bins=100)
    # plt.show()
    #
    # plt.hist(strain_incre_gamma, bins=100)
    # plt.show()

    kashk = 1
    return strain_incre_all, strain_mag
