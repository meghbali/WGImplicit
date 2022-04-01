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
import pickle


def mean(inp):
    output = (inp[0][0] + inp[1][0] + inp[2][0]) / 3.0
    return output


def inv(inp):
    inp_ten = sixtothree(inp)
    inp_ten_inv = np.linalg.inv(inp_ten)
    out = threetosix(inp_ten_inv)
    return out


def sixtothree(inp):
    out = np.zeros((3, 3))
    out[0][0] = inp[0][0]
    out[1][1] = inp[1][0]
    out[2][2] = inp[2][0]
    out[0][1] = inp[5][0] / np.sqrt(2.0)
    out[0][2] = inp[4][0] / np.sqrt(2.0)
    out[1][2] = inp[3][0] / np.sqrt(2.0)
    out[1][0] = out[0][1]
    out[2][0] = out[0][2]
    out[2][1] = out[1][2]
    return out


def threetosix(inp):
    out = np.zeros((6, 1))
    out[0][0] = inp[0][0]
    out[1][0] = inp[1][1]
    out[2][0] = inp[2][2]
    out[3][0] = inp[1][2] * np.sqrt(2.0)
    out[4][0] = inp[0][2] * np.sqrt(2.0)
    out[5][0] = inp[0][1] * np.sqrt(2.0)
    return out


def deviator(inp):
    p = mean(inp)
    s = np.array([[inp[0][0] - p], [inp[1][0] - p], [inp[2][0] - p], [inp[3][0]], [inp[4][0]], [inp[5][0]]])
    return s


def norm(inp):
    out = 0.0
    for i in range(6):
        out += inp[i][0] ** 2.0

    out = np.sqrt(out)
    return out


def dot(inp1, inp2):
    out = 0.0
    for i in range(6):
        out += inp1[i][0] * inp2[i][0]

    return out


def starinvf(inp, f):
    f_ten = sixtothree(f.value)
    inp_ten1 = sixtothree(inp)
    inp_ten2 = np.matmul(inp_ten1, np.linalg.inv(f_ten))
    out = threetosix(inp_ten2)
    return out


def starf(inp, f):
    f_ten = sixtothree(f.value)
    inp_ten1 = sixtothree(inp)
    inp_ten2 = np.matmul(inp_ten1, f_ten)
    out = threetosix(inp_ten2)
    return out


def j2(inp):
    s = deviator(inp)
    out = 0.0
    for i in range(6):
        out += s[i][0] ** 2.0 / 2.0
    return out


def j3(inp):
    inp_s = deviator(inp)
    inp_s_ten = sixtothree(inp_s)
    out = np.linalg.det(inp_s_ten)
    return out


def lode(inp):
    j2_val = j2(inp)
    j3_val = j3(inp)
    if j2_val == 0.0:
        out = 1.0
    else:
        out = (j3_val / 2.0) * (3.0 / j2_val) ** 1.5
    return out


def eq(inp):
    j2_val = j2(inp)
    out = np.sqrt(3.0 * j2_val)
    return out


def eqf(inp, f):
    inp_star = starinvf(inp, f)
    j2_val = j2(inp_star)
    out = np.sqrt(3.0 * j2_val)
    return out


def ss(inp):
    inp_s = deviator(inp)
    inp_s_ten = sixtothree(inp_s)
    a_ten = np.matmul(inp_s_ten, inp_s_ten)
    out = threetosix(a_ten)
    return out


def invstarinvf(inp, f):
    inp_star_1 = starinvf(inp, f)
    inp_star_1_ten = sixtothree(inp_star_1)
    inv_inp_star_1_ten = np.linalg.inv(inp_star_1_ten)
    out = threetosix(inv_inp_star_1_ten)
    return out


def invstarinvfn(inp, f):
    inp_inv_star_1 = invstarinvf(inp.value, f)
    inp_inv_star_1_ten = sixtothree(inp_inv_star_1)
    inp_star_1 = starinvf(inp.value_step, f)
    inp_star_1_ten = sixtothree(inp_star_1)
    out_1 = np.matmul(inp_inv_star_1_ten, inp_star_1_ten)
    out = threetosix(out_1)
    return out


def data_loader_dat(file_name):
    get_data = pickle.load(open(file_name, "rb"))
    return get_data


def data_dumper_dat(file_name, outputset):
    pickle.dump(outputset, open(file_name, "wb"))
    return
