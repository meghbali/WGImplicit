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
import wgimplicit_classes as cls
import wgimplicit_module_main as main
import wgimplicit_module_utility as util
import os
import time


def mainfunc(kpin, kvrin, nrepeat, nextrepeat):
    np.seterr(all='print')

    # === define sample
    # see the definitions in the "Sample" class
    # when fabric is isotropic, chi=0 (no fabric evolution), x == b_0, a == a_f WG_fabric_noncoaxial reduces to
    # WG_noncoaxial. when lmbd = 0 WG_fabric_noncoaxial reduces to WG_fabric.
    # when both conditions are met the code reduces to WG_basic
    # for WG_fabric and WG_fabric_noncoaxial model, the cap should be turned off.

    ottawa_sand = \
        cls.Sample(p0=1.0, mu=0.8, nu=0.3, g0=900.0, vrc0=0.74, ncr=0.4, hcr=565685.424949, sinphicv=0.53, a=0.008,
                   beta=1.3, alpha=1.5, x=0.007, chi=0.0, hic=0.005, nic=0.4, cap=0, a_f=0.008, b_0=0.008, n_1=0.0,
                   n_2=0.0, lmbd=0.0)

    ottawa_sand2 = \
        cls.Sample(p0=1.0, mu=0.8, nu=0.3, g0=900.0, vrc0=0.74, ncr=0.4, hcr=565685.424949, sinphicv=0.53, a=0.008,
                   beta=1.3, alpha=1.5, x=0.007, chi=0.0, hic=0.01, nic=0.4, cap=0, a_f=0.008, b_0=0.008, n_1=0.0,
                   n_2=0.0, lmbd=0.0)

    # set the sample to be used for the analysis
    sample = ottawa_sand2

    # ===  set solver parameters
    # see the definitions in the "Solution" class. User usually does not need to change these.
    tolerance_elastic = 1E-06
    tolerance_return_mapping = 1E-06
    error_initial = 1E06
    maximum_iterations_elastic = 100
    maximum_iterations_return_mapping = 100
    print_step_information = True
    print_frequency = 1000
    step_pause = 0.0001
    solver = cls.Solver(tol_elas=tolerance_elastic, tol_rtmp=tolerance_return_mapping, err_in=error_initial,
                        maxit_elas=maximum_iterations_elastic, maxit_rtmp=maximum_iterations_return_mapping,
                        stpinfo=print_step_information, freq=print_frequency, stp_pause=step_pause)

    # ===  multiple runs with reset
    # n_pin, pin_min, pin_max = 10, 50, 500
    n_pin, pin_min, pin_max = 1, 375, 375
    pin = np.random.rand(n_pin, 1) * (pin_max - pin_min) + pin_min
    # pin = np.array([[50.0], [100.0], [150.0], [200.0], [250.0], [300.0], [350.0], [400.0], [450.0], [500.0]])

    # n_vrin, vrin_min, vrin_max = 10, 0.5, 0.74
    n_vrin, vrin_min, vrin_max = 1, 0.64, 0.64
    vrin = np.random.rand(n_vrin, 1) * (vrin_max - vrin_min) + vrin_min
    # vrin = np.array([[0.5], [0.53], [0.56], [0.58], [0.61], [0.63], [0.66], [0.69], [0.71], [0.74]])

    nrepeat = 1

    kpin = 0
    krepeat = nextrepeat

    for ipin in pin[kpin:(kpin + 1), 0]:  # loop over initial confining stresses
        kpin += 1
        kvrin = 0
        for ivrin in vrin[kvrin:(kvrin + 1), 0]:  # loop over initial void ratios
            kvrin += 1

            for alpha in [-1.5]:  # loop over different alphas for proportional strain paths
                # repeat from beginning
                irepeat = 0
                # for irepeat in range(nrepeat):
                while irepeat < nrepeat:
                    print("====== CONFINING " + str(kpin))
                    print("====== VOID RATIO " + str(kvrin))
                    print("====== REPEAT " + str(krepeat))
                    time.sleep(1.0)

                    # ===  initialize state variables

                    # main input block
                    confining_initial = ipin  # initial confining stress (kPa)
                    vr_initial = ivrin  # initial void ratio
                    fabric_initial = [1.0, 1.0, 1.0]  # initial fabric in principal space
                    bedding_plane = 0.0  # rotation around axis 2 (fabric)
                    p_angle = 0.0  # rotation around axis 3 (fabric)

                    # initialize some variables
                    stress_initial, fabric_global_initial, strain_plastic_initial \
                        = main.initialize(bdplane=bedding_plane, pangle=p_angle, f_in=fabric_initial,
                                          conf=confining_initial, smp=sample)

                    # set initial state
                    void_ratio, fabric, stress, strain, strain_pl, strain_pl_cap \
                        = main.set_in_state(vr_in=vr_initial, strsin=stress_initial, fin=fabric_global_initial,
                                            strnin=strain_plastic_initial, strnpin=strain_plastic_initial,
                                            strnpcin=strain_plastic_initial)

                    # ===  generate random strain increment
                    if_generate = True

                    if if_generate:

                        # if os.path.exists('strain_increments_' + str(kpin) + '_' + str(kvrin) + '_'
                        #                   + str(irepeat + nextrepeat) + '.dat'):
                        #     os.remove('strain_increments_' + str(kpin) + '_' + str(kvrin)
                        #               + '_' + str(irepeat + nextrepeat) + '.dat')

                        maximum_strain_increment_magnitude = 4.0 * 0.0004
                        strain_increments, strain_mag \
                            = main.random_strain_increment(strain_mag_max=maximum_strain_increment_magnitude,
                                                           strain_incre_number=200)

                        # util.data_dumper_dat(file_name='strain_increments_'
                        #                                + str(kpin) + '_' + str(kvrin) + '_' +
                        #                                str(irepeat + nextrepeat) + '.dat',
                        #                                outputset=strain_increments)
                    else:
                        strain_increments = util.data_loader_dat(file_name='strain_increments_'
                                                                           + str(kpin) + '_' + str(kvrin) + '_'
                                                                           + str(irepeat + nextrepeat) + '.dat')

                    # ================================ main solution loop
                    print("========== START OF ANALYSIS ==========")

                    # initialize recorder
                    mystep = cls.Solution(strain=strain, stress=stress, strain_pl=strain_pl, strain_pl_c=strain_pl_cap,
                                          vr=void_ratio, f=fabric)

                    step_total = 0

                    flagplas2 = 0
                    # stop-continue
                    # for iload in range(strain_increments.shape[0]):
                    # for iload in range(strain_mag.shape[0]):
                    for iload in range(1):

                        # ==================== define loading
                        # see the definitions in the "Load" class
                        loading_goal = [1, 1, 1, 1, 1, 1]  # general strain controlled
                        # loading_goal = [0, 0, 1, 0, 0, 0]  # uniaxial

                        # general strain controlled
                        # strain_goal = np.transpose(strain_increments[iload: (iload + 1), :])
                        # strain_goal = np.array([[-strain_mag[iload, 0]/2], [-strain_mag[iload, 0]/2],
                        #                         [strain_mag[iload, 0]], [0.0], [0.0], [0.0]])  # custom
                        strain_goal = np.array([[0.07 / alpha], [0.07 / alpha], [0.07], [0.0], [0.0], [0.0]])  # custom

                        number_of_steps = 100

                        # if iload == 0:
                        #     strain_goal = np.array([[-0.06 / 2.0], [-0.06 / 2.0], [0.06],
                        #                             [0.0], [0.0], [0.0]])  # custom
                        #     number_of_steps = 1000
                        # elif iload == 1:
                        #     strain_goal = np.array([[(0.06 / 2.0) / 14.0], [(0.06 / 2.0) / 14.0], [-0.06 / 14.0],
                        #                             [0.0], [0.0], [0.0]])  # custom
                        #     number_of_steps = 70
                        # else:
                        #     strain_goal = np.array([[-0.01 / 2.0 + (-0.06 / 2.0) / 14.0],
                        #                             [-0.01 / 2.0 + (-0.06 / 2.0) / 14.0], [0.01 + 0.06 / 14.0],
                        #                             [0.0], [0.0], [0.0]])  # custom
                        #     number_of_steps = 240

                        stress_goal = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

                        # create the load object
                        load = cls.Load(goal=loading_goal, nstep=number_of_steps, strain=strain_goal,
                                        stress=stress_goal)

                        # reach the set loading goal
                        mystep, void_ratio, fabric, strain, stress, strain_pl, strain_pl_cap, flagplas \
                            = main.reach_goal(load, solver, sample, void_ratio, stress, strain, strain_pl,
                                              strain_pl_cap, fabric, mystep, step_total)

                        if flagplas == 1:
                            flagplas2 = 1

                        # initialize for next goal
                        void_ratio, fabric, stress, strain, strain_pl, strain_pl_cap \
                            = main.set_in_state(vr_in=void_ratio.value, strsin=stress.value, fin=fabric.value,
                                                strnin=strain.value, strnpin=strain_pl.value,
                                                strnpcin=strain_pl_cap.value)

                        step_total += load.nstep
                        kflag = 0
                        if abs(mystep.p[-1, 0]) < 1.0 or mystep.q[-1, 0] < 0.1:
                            if iload < 25:
                                kflag = 1

                            print('p and q close to zero. Terminate.')
                            break

                    # if kflag == 0 and flagplas2 == 0:
                    if flagplas2 == 0:
                        if os.path.exists('output_alpha_' + str(kpin) + '_' + str(kvrin) + '_'
                                          + str(krepeat) + '_' + str(alpha) + '.dat'):
                            os.remove('output_alpha_' + str(kpin) + '_' + str(kvrin) + '_' + str(krepeat) + '_'
                                      + str(alpha) + '.dat')

                        util.data_dumper_dat(file_name='output_alpha_' + str(kpin) + '_' + str(kvrin) + '_' +
                                                       str(krepeat) + '_' + str(alpha) + '.dat',
                                             outputset=[mystep.stress, mystep.strain, mystep.strain_pl,
                                                        mystep.strain_pl_c, mystep.vr, mystep.p, mystep.q,
                                                        mystep.gamma_pl, mystep.gamma, mystep.ksecant])
                        krepeat += 1
                        irepeat += 1
