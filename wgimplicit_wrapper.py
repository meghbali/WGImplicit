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
import wgimplicit_main as main
import os

"""
this wrapper is for batch run of the main code in wgimplicit_main. Different parameters of the model have to be tuned 
inside that code.
"""

print("wrapper")
folder = os.getcwd()
noutput = 0
for filename in os.listdir(folder):
    if 'output' in filename:
        noutput += 1

which_pin = 0
which_vrin = 0
completed_rep = noutput

total_repeat = 0
main.mainfunc(kpin=which_pin, kvrin=which_vrin, nrepeat=total_repeat - completed_rep, nextrepeat=completed_rep)
