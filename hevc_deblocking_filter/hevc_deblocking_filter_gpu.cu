/*
Copyright (c) 2019 Roman Kazantsev
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <omp.h>
#include <chrono>


using namespace std;

const int sample_block_size = 8;

unsigned int _width;
unsigned int _height;
unsigned int _new_width;
unsigned int _new_height;

unsigned int _chroma_width;
unsigned int _chroma_height;
unsigned int _new_chroma_width;
unsigned int _new_chroma_height;

// strenght of vertical and horizontal boundaries for luma
unsigned int _num_vert_bs;
unsigned int _num_hor_bs;

// strength of vertical and horizontal boundaries for chroma components
unsigned int _num_chroma_vert_bs;
unsigned int _num_chroma_hor_bs;

unsigned int _Qp; // quantization parameter

				  // GPU buffers for Y, U, V
unsigned char *_gpu_Y_ptr;
unsigned char *_gpu_U_ptr;
unsigned char *_gpu_V_ptr;

// GPU buffers for arrays with boundary stregths
unsigned char *_gpu_vert_bs;
unsigned char *_gpu_hor_bs;
unsigned char *_gpu_chroma_vert_bs;
unsigned char *_gpu_chroma_hor_bs;

// CPU buffers
unsigned char *Y_pinned_ptr;
unsigned char *U_pinned_ptr;
unsigned char *V_pinned_ptr;

// boundary strengths
unsigned char *_vert_bs;
unsigned char *_hor_bs;
unsigned char *_chroma_vert_bs;
unsigned char *_chroma_hor_bs;

__device__ unsigned int GetBeta(unsigned int QP) {
	unsigned int beta_table[52] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // QP 0..15
		6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, // QP 16..31
		26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, // QP 32 .. 47
		58, 60, 62, 64 // QP 48..51
	};

	if (QP > 51) return beta_table[51];
	return beta_table[QP];
}

__device__ unsigned int GetTc(unsigned int QP) {
	unsigned int tc_table[52] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // QP 0..15
		0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, // QP 16..31
		3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, // QP 32..47
		14, 16, 18, 20  // QP 48..51
	};

	if (QP > 51) return tc_table[51];
	return tc_table[QP];
}

__device__ bool CheckLocalAdaptivity(
	int p00, int p01, int p02,
	int p30, int p31, int p32,
	int q00, int q01, int q02,
	int q30, int q31, int q32,
	int beta) {
	// p03, p02, p01, p00 | q00, q01, q02, q03
	// p13, p12, p11, p10 | q10, q11, q12, q13
	// p23, p22, p21, p00 | q20, q21, q22, q23
	// p33, p32, p31, p30 | q30, q31, q32, q33

	// condition (1)
	if ((abs(p02 - 2 * p01 + p00) + abs(p32 - 2 * p31 + p30) +
		abs(q02 - 2 * q01 + q00) + abs(q32 - 2 * q31 + q30)) < beta) return true;
	return false;
}

__device__ bool IsStrongFilterToUse(
	int p00, int p01, int p02, int p03,
	int p30, int p31, int p32, int p33,
	int q00, int q01, int q02, int q03,
	int q30, int q31, int q32, int q33,
	int beta, int tc) {
	if (((abs(p02 - 2 * p01 + p00) + abs(q02 - 2 * q01 + q00)) < beta / 8) &&
		((abs(p32 - 2 * p31 + p30) + abs(q32 - 2 * q31 + q30)) < beta / 8) &&
		((abs(p03 - p00) + abs(q00 - q03)) < beta / 8) &&
		((abs(p33 - p30) + abs(q30 - q33)) < beta / 8) &&
		(abs(p00 - q00) < 5 * tc / 2) &&
		(abs(p30 - q30) < 5 * tc / 2)) return true;
	return false;
}

__device__ bool AreP0P1Modified(
	int p00, int p10, int p20, int p30,
	int p03, int p13, int p23, int p33,
	int beta) {
	if ((abs(p20 - 2 * p10 + p00) + abs(p23 - 2 * p13 + p03)) < 3 * beta / 16)
		return true;
	return false;
}

// clip value so that it will be in a range [-c; c]
__device__ int Clip1(int delta, int c) {
	return min(max(-c, delta), c);
}

// clip value so that it will be in a range [0; c]
__device__ int Clip2(int value, int c) {
	return min(max(0, value), c);
}

__device__ void ApplyStrongFilter(
	unsigned char *p00, unsigned char *p01, unsigned char *p02, unsigned char *p03,
	unsigned char *p10, unsigned char *p11, unsigned char *p12, unsigned char *p13,
	unsigned char *p20, unsigned char *p21, unsigned char *p22, unsigned char *p23,
	unsigned char *p30, unsigned char *p31, unsigned char *p32, unsigned char *p33,

	unsigned char *q00, unsigned char *q01, unsigned char *q02, unsigned char *q03,
	unsigned char *q10, unsigned char *q11, unsigned char *q12, unsigned char *q13,
	unsigned char *q20, unsigned char *q21, unsigned char *q22, unsigned char *q23,
	unsigned char *q30, unsigned char *q31, unsigned char *q32, unsigned char *q33,

	unsigned int QP
) {
	int max_v = (1 << 8) - 1;

	// p03, p02, p01, p00 | q00, q01, q02, q03
	// p13, p12, p11, p10 | q10, q11, q12, q13
	// p23, p22, p21, p00 | q20, q21, q22, q23
	// p33, p32, p31, p30 | q30, q31, q32, q33
	int _p0, _p1, _p2, _p3, _q0, _q1, _q2, _q3;
	_p0 = *p00; _p1 = *p01; _p2 = *p02; _p3 = *p03; _q0 = *q00; _q1 = *q01; _q2 = *q02; _q3 = *q03;
	// compute deltas for P block
	int delta0_p = (_p2 + 2 * _p1 - 6 * _p0 + 2 * _q0 + _q1 + 4) >> 3; // δ0s = (p2 + 2p1 − 6p0 + 2q0 + q1 + 4) >> 3
	int delta1_p = (_p2 - 3 * _p1 + _p0 + _q0 + 2) >> 2; // δ1s = (p2 − 3p1 + p0 + q0 + 2) >> 2
	int delta2_p = (2 * _p3 - 5 * _p2 + _p1 + _p0 + _q0 + 4) >> 3; // δ2s = (2p3 − 5p2 + p1 + p0 + q0 + 4) >> 3

	// compute deltas for Q block
	int delta0_q = (_q2 + 2 * _q1 - 6 * _q0 + 2 * _p0 + _p1 + 4) >> 3; // δ0s = (q2 + 2q1 − 6q0 + 2p0 + p1 + 4) >> 3
	int delta1_q = (_q2 - 3 * _q1 + _q0 + _p0 + 2) >> 2; // δ1s = (q2 − 3q1 + q0 + p0 + 2) >> 2
	int delta2_q = (2 * _q3 - 5 * _q2 + _q1 + _q0 + _p0 + 4) >> 3; // δ2s = (2q3 − 5q2 + q1 + q0 + p0 + 4) >> 3

	// clip deltas for P block
	int c = 2 * GetTc(QP);
	delta0_p = Clip1(delta0_p, c); delta1_p = Clip1(delta1_p, c); delta2_p = Clip1(delta2_p, c);
	// clip deltas for Q block
	delta0_q = Clip1(delta0_q, c); delta1_q = Clip1(delta1_q, c); delta2_q = Clip1(delta2_q, c);

	// filter and clip values for P block
	*p00 = Clip2(_p0 + delta0_p, max_v); *p01 = Clip2(_p1 + delta1_p, max_v); *p02 = Clip2(_p2 + delta2_p, max_v);
	// filter and clip values for Q block
	*q00 = Clip2(_q0 + delta0_q, max_v); *q01 = Clip2(_q1 + delta1_q, max_v); *q02 = Clip2(_q2 + delta2_q, max_v);

	_p0 = *p10; _p1 = *p11; _p2 = *p12; _p3 = *p13; _q0 = *q10; _q1 = *q11; _q2 = *q12; _q3 = *q13;
	// compute deltas for P block
	delta0_p = (_p2 + 2 * _p1 - 6 * _p0 + 2 * _q0 + _q1 + 4) >> 3; // δ0s = (p2 + 2p1 − 6p0 + 2q0 + q1 + 4) >> 3
	delta1_p = (_p2 - 3 * _p1 + _p0 + _q0 + 2) >> 2; // δ1s = (p2 − 3p1 + p0 + q0 + 2) >> 2
	delta2_p = (2 * _p3 - 5 * _p2 + _p1 + _p0 + _q0 + 4) >> 3; // δ2s = (2p3 − 5p2 + p1 + p0 + q0 + 4) >> 3

	// compute deltas for Q block
	delta0_q = (_q2 + 2 * _q1 - 6 * _q0 + 2 * _p0 + _p1 + 4) >> 3; // δ0s = (q2 + 2q1 − 6q0 + 2p0 + p1 + 4) >> 3
	delta1_q = (_q2 - 3 * _q1 + _q0 + _p0 + 2) >> 2; // δ1s = (q2 − 3q1 + q0 + p0 + 2) >> 2
	delta2_q = (2 * _q3 - 5 * _q2 + _q1 + _q0 + _p0 + 4) >> 3; // δ2s = (2q3 − 5q2 + q1 + q0 + p0 + 4) >> 3

	// clip deltas for P block
	delta0_p = Clip1(delta0_p, c); delta1_p = Clip1(delta1_p, c); delta2_p = Clip1(delta2_p, c);
	// clip deltas for Q block
	delta0_q = Clip1(delta0_q, c); delta1_q = Clip1(delta1_q, c); delta2_q = Clip1(delta2_q, c);

	// filter and clip values for P block
	*p10 = Clip2(_p0 + delta0_p, max_v); *p11 = Clip2(_p1 + delta1_p, max_v); *p12 = Clip2(_p2 + delta2_p, max_v);
	// filter and clip values for Q block
	*q10 = Clip2(_q0 + delta0_q, max_v); *q11 = Clip2(_q1 + delta1_q, max_v); *q12 = Clip2(_q2 + delta2_q, max_v);

	_p0 = *p20; _p1 = *p21; _p2 = *p22; _p3 = *p23; _q0 = *q20; _q1 = *q21; _q2 = *q22; _q3 = *q23;
	// compute deltas for P block
	delta0_p = (_p2 + 2 * _p1 - 6 * _p0 + 2 * _q0 + _q1 + 4) >> 3; // δ0s = (p2 + 2p1 − 6p0 + 2q0 + q1 + 4) >> 3
	delta1_p = (_p2 - 3 * _p1 + _p0 + _q0 + 2) >> 2; // δ1s = (p2 − 3p1 + p0 + q0 + 2) >> 2
	delta2_p = (2 * _p3 - 5 * _p2 + _p1 + _p0 + _q0 + 4) >> 3; // δ2s = (2p3 − 5p2 + p1 + p0 + q0 + 4) >> 3

															   // compute deltas for Q block
	delta0_q = (_q2 + 2 * _q1 - 6 * _q0 + 2 * _p0 + _p1 + 4) >> 3; // δ0s = (q2 + 2q1 − 6q0 + 2p0 + p1 + 4) >> 3
	delta1_q = (_q2 - 3 * _q1 + _q0 + _p0 + 2) >> 2; // δ1s = (q2 − 3q1 + q0 + p0 + 2) >> 2
	delta2_q = (2 * _q3 - 5 * _q2 + _q1 + _q0 + _p0 + 4) >> 3; // δ2s = (2q3 − 5q2 + q1 + q0 + p0 + 4) >> 3

															   // clip deltas for P block
	delta0_p = Clip1(delta0_p, c); delta1_p = Clip1(delta1_p, c); delta2_p = Clip1(delta2_p, c);
	// clip deltas for Q block
	delta0_q = Clip1(delta0_q, c); delta1_q = Clip1(delta1_q, c); delta2_q = Clip1(delta2_q, c);
	// filter and clip values for P block
	*p20 = Clip2(_p0 + delta0_p, max_v); *p21 = Clip2(_p1 + delta1_p, max_v); *p22 = Clip2(_p2 + delta2_p, max_v);
	// filter and clip values for Q block
	*q20 = Clip2(_q0 + delta0_q, max_v); *q21 = Clip2(_q1 + delta1_q, max_v); *q22 = Clip2(_q2 + delta2_q, max_v);

	_p0 = *p30; _p1 = *p31; _p2 = *p32; _p3 = *p33; _q0 = *q30; _q1 = *q31; _q2 = *q32; _q3 = *q33;
	// compute deltas for P block
	delta0_p = (_p2 + 2 * _p1 - 6 * _p0 + 2 * _q0 + _q1 + 4) >> 3; // δ0s = (p2 + 2p1 − 6p0 + 2q0 + q1 + 4) >> 3
	delta1_p = (_p2 - 3 * _p1 + _p0 + _q0 + 2) >> 2; // δ1s = (p2 − 3p1 + p0 + q0 + 2) >> 2
	delta2_p = (2 * _p3 - 5 * _p2 + _p1 + _p0 + _q0 + 4) >> 3; // δ2s = (2p3 − 5p2 + p1 + p0 + q0 + 4) >> 3

	// compute deltas for Q block
	delta0_q = (_q2 + 2 * _q1 - 6 * _q0 + 2 * _p0 + _p1 + 4) >> 3; // δ0s = (q2 + 2q1 − 6q0 + 2p0 + p1 + 4) >> 3
	delta1_q = (_q2 - 3 * _q1 + _q0 + _p0 + 2) >> 2; // δ1s = (q2 − 3q1 + q0 + p0 + 2) >> 2
	delta2_q = (2 * _q3 - 5 * _q2 + _q1 + _q0 + _p0 + 4) >> 3; // δ2s = (2q3 − 5q2 + q1 + q0 + p0 + 4) >> 3

	// clip deltas for P block
	delta0_p = Clip1(delta0_p, c); delta1_p = Clip1(delta1_p, c); delta2_p = Clip1(delta2_p, c);
	// clip deltas for Q block
	delta0_q = Clip1(delta0_q, c); delta1_q = Clip1(delta1_q, c); delta2_q = Clip1(delta2_q, c);

	// filter and clip values for P block
	*p30 = Clip2(_p0 + delta0_p, max_v); *p31 = Clip2(_p1 + delta1_p, max_v); *p32 = Clip2(_p2 + delta2_p, max_v);
	// filter and clip values for Q block
	*q30 = Clip2(_q0 + delta0_q, max_v); *q31 = Clip2(_q1 + delta1_q, max_v); *q32 = Clip2(_q2 + delta2_q, max_v);
}

__device__ void ApplyNormalFilter(
	unsigned char *p00, unsigned char *p01, unsigned char *p02,
	unsigned char *p10, unsigned char *p11, unsigned char *p12,
	unsigned char *p20, unsigned char *p21, unsigned char *p22,
	unsigned char *p30, unsigned char *p31, unsigned char *p32,

	unsigned char *q00, unsigned char *q01, unsigned char *q02,
	unsigned char *q10, unsigned char *q11, unsigned char *q12,
	unsigned char *q20, unsigned char *q21, unsigned char *q22,
	unsigned char *q30, unsigned char *q31, unsigned char *q32,

	unsigned int QP
) {
	int max_v = (1 << 8) - 1;
	// p03, p02, p01, p00 | q00, q01, q02, q03
	// p13, p12, p11, p10 | q10, q11, q12, q13
	// p23, p22, p21, p00 | q20, q21, q22, q23
	// p33, p32, p31, p30 | q30, q31, q32, q33
	unsigned int tc = GetTc(QP);
	unsigned int beta = GetBeta(QP);
	int c = 2 * tc;
	int c2 = tc / 2;

	int _p00 = *p00, _p01 = *p01, _p02 = *p02, _q00 = *q00, _q01 = *q01, _q02 = *q02;
	int _p10 = *p10, _p11 = *p11, _p12 = *p12, _q10 = *q10, _q11 = *q11, _q12 = *q12;
	int _p20 = *p20, _p21 = *p21, _p22 = *p22, _q20 = *q20, _q21 = *q21, _q22 = *q22;
	int _p30 = *p30, _p31 = *p31, _p32 = *p32, _q30 = *q30, _q31 = *q31, _q32 = *q32;

	bool cond5 = false;
	// | p2,0 − 2p1,0 + p0,0| + |p2,3 − 2p1,3 + p0,3| < 3/16β
	if (std::abs(_p02 - 2 * _p01 + _p00) + std::abs(_p32 - 2 * _p31 + _p30) < 3 * beta / 16) cond5 = true;

	bool cond6 = false;
	// |q2,0 − 2q1,0 + q0,0| + |q2,3 − 2q1,3 + q0,3| < 3/16β
	if (std::abs(_q02 - 2 * _q01 + _q00) + std::abs(_q32 - 2 * _q31 + _q30) < 3 * beta / 16) cond6 = true;

	// handle the 0-th row
	// δ0 = (9(q0 − p0) − 3(q1 − p1) + 8) >> 4.
	int delta00 = (9 * (_q00 - _p00) - 3 * (_q01 - _p01) + 8) >> 4;
	if (std::abs(delta00) < 10 * tc) {
		// filter p00
		int Delta00 = Clip1(delta00, c);
		// δp1 = (((p2 + p0 + 1) >> 1) − p1 + 0) >> 1
		//int delta_p01 = (((_p02 + _p00 + 1) >> 1) - _p01 + Delta00) >> 1;
		int delta_p01 = (((_p02 + _p00 + 1) >> 1) - _p01 + Delta00) >> 1;
		int Delta_p01 = Clip1(delta_p01, c2);

		// δq1 = (((q2 + q0 + 1) >> 1) − q1 − 0) >> 1
		//int delta_q01 = (((_q02 + _q00 + 1) >> 1) - _q01 - Delta00) >> 1;
		int delta_q01 = (((_q02 + _q00 + 1) >> 1) - _q01 - Delta00) >> 1;
		int Delta_q01 = Clip1(delta_q01, c2);

		// filter p00 and q00
		*p00 = Clip2(_p00 + Delta00, max_v);
		*q00 = Clip2(_q00 - Delta00, max_v);

		// filter p01
		if (cond5) *p01 = Clip2(_p01 + Delta_p01, max_v);

		// filter q01
		if (cond6) *q01 = Clip2(_q01 + Delta_q01, max_v);
	}

	// handle the 1-st row
	int delta10 = (9 * (_q10 - _p10) - 3 * (_q11 - _p11) + 8) >> 4;
	if (std::abs(delta10) < 10 * tc) {
		// filter p10
		int Delta10 = Clip1(delta10, c);
		// δp1 = (((p2 + p0 + 1) >> 1) − p1 + 0) >> 1
		//int delta_p11 = (((_p12 + _p10 + 1) >> 1) - _p11 + Delta10) >> 1;
		int delta_p11 = (((_p12 + _p10 + 1) >> 1) - _p11 + Delta10) >> 1;
		int Delta_p11 = Clip1(delta_p11, c2);

		// δq1 = (((q2 + q0 + 1) >> 1) − q1 − 0) >> 1
		//int delta_q11 = (((_q12 + _q10 + 1) >> 1) - _q11 - Delta10) >> 1;
		int delta_q11 = (((_q12 + _q10 + 1) >> 1) - _q11 - Delta10) >> 1;
		int Delta_q11 = Clip1(delta_q11, c2);

		// filter p10 and q10
		*p10 = Clip2(_p10 + Delta10, max_v);
		*q10 = Clip2(_q10 - Delta10, max_v);

		// filter p11
		if (cond5) *p11 = Clip2(_p11 + Delta_p11, max_v);

		// filter q11
		if (cond6) *q11 = Clip2(_q11 + Delta_q11, max_v);
	}

	// handle the 2-nd row
	int delta20 = (9 * (_q20 - _p20) - 3 * (_q21 - _p21) + 8) >> 4;
	if (std::abs(delta20) < 10 * tc) {
		// filter p20
		int Delta20 = Clip1(delta20, c);
		// δp1 = (((p2 + p0 + 1) >> 1) − p1 + 0) >> 1
		//int delta_p21 = (((_p22 + _p20 + 1) >> 1) - _p21 + Delta20) >> 1;
		int delta_p21 = (((_p22 + _p20 + 1) >> 1) - _p21 + Delta20) >> 1;
		int Delta_p21 = Clip1(delta_p21, c2);

		// δq1 = (((q2 + q0 + 1) >> 1) − q1 − 0) >> 1
		//int delta_q21 = (((_q22 + _q20 + 1) >> 1) - _q21 - Delta20) >> 1;
		int delta_q21 = (((_q22 + _q20 + 1) >> 1) - _q21 - Delta20) >> 1;
		int Delta_q21 = Clip1(delta_q21, c2);

		// filter p20 and q20
		*p20 = Clip2(_p20 + Delta20, max_v);
		*q20 = Clip2(_q20 - Delta20, max_v);

		// filter p21
		if (cond5) *p21 = Clip2(_p21 + Delta_p21, max_v);

		// filter q21
		if (cond6) *q21 = Clip2(_q21 + Delta_q21, max_v);
	}

	// handle the 3-rd row
	int delta30 = (9 * (_q30 - _p30) - 3 * (_q31 - _p31) + 8) >> 4;
	if (std::abs(delta30) < 10 * tc) {
		// filter p30
		int Delta30 = Clip1(delta30, c);
		// δp1 = (((p2 + p0 + 1) >> 1) − p1 + 0) >> 1
		//int delta_p31 = (((_p32 + _p30 + 1) >> 1) - _p31 + Delta20) >> 1;
		int delta_p31 = (((_p32 + _p30 + 1) >> 1) - _p31 + Delta30) >> 1;
		int Delta_p31 = Clip1(delta_p31, c2);

		// δq1 = (((q2 + q0 + 1) >> 1) − q1 − 0) >> 1
		//int delta_q31 = (((_q32 + _q30 + 1) >> 1) - _q31 - Delta30) >> 1;
		int delta_q31 = (((_q32 + _q30 + 1) >> 1) - _q31 - Delta30) >> 1;
		int Delta_q31 = Clip1(delta_q31, c2);

		// filter p30 and q30
		*p30 = Clip2(_p30 + Delta30, max_v);
		*q30 = Clip2(_q30 - Delta30, max_v);

		// filter p31
		if (cond5) *p31 = Clip2(_p31 + Delta_p31, max_v);

		// filter q31
		if (cond6) *q31 = Clip2(_q31 + Delta_q31, max_v);
	}

	return;
}

__device__ void DeblockingFilterLuma(
	unsigned char *p00, unsigned char *p01, unsigned char *p02, unsigned char *p03,
	unsigned char *p10, unsigned char *p11, unsigned char *p12, unsigned char *p13,
	unsigned char *p20, unsigned char *p21, unsigned char *p22, unsigned char *p23,
	unsigned char *p30, unsigned char *p31, unsigned char *p32, unsigned char *p33,

	unsigned char *q00, unsigned char *q01, unsigned char *q02, unsigned char *q03,
	unsigned char *q10, unsigned char *q11, unsigned char *q12, unsigned char *q13,
	unsigned char *q20, unsigned char *q21, unsigned char *q22, unsigned char *q23,
	unsigned char *q30, unsigned char *q31, unsigned char *q32, unsigned char *q33,
	unsigned int beta, unsigned int tc, unsigned int _Qp) {
	// check local adaptivity - condition 1
	bool cond1 = CheckLocalAdaptivity(
		*p00, *p01, *p02,
		*p30, *p31, *p32,
		*q00, *q01, *q02,
		*q30, *q31, *q32,
		beta);

	// check if to use strong filter
	bool is_strong_filter_to_use = false;
	if (cond1) {
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		is_strong_filter_to_use = IsStrongFilterToUse(
			*p00, *p01, *p02, *p03,
			*p30, *p31, *p32, *p33,
			*q00, *q01, *q02, *q03,
			*q30, *q31, *q32, *q33,
			beta, tc);
	}

	// strong filter
	if (cond1 && is_strong_filter_to_use) {
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		ApplyStrongFilter(
			p00, p01, p02, p03,
			p10, p11, p12, p13,
			p20, p21, p22, p23,
			p30, p31, p32, p33,
			q00, q01, q02, q03,
			q10, q11, q12, q13,
			q20, q21, q22, q23,
			q30, q31, q32, q33,
			_Qp
		);
	}

	// normal filter
	if (cond1 && !is_strong_filter_to_use) {
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		ApplyNormalFilter(
			p00, p01, p02,
			p10, p11, p12,
			p20, p21, p22,
			p30, p31, p32,
			q00, q01, q02,
			q10, q11, q12,
			q20, q21, q22,
			q30, q31, q32,
			_Qp);
	}
}

__device__ void DeblockingFilterChroma(
	unsigned char *p00, unsigned char *p01,
	unsigned char *p10, unsigned char *p11,
	unsigned char *p20, unsigned char *p21,
	unsigned char *p30, unsigned char *p31,

	unsigned char *q00, unsigned char *q01,
	unsigned char *q10, unsigned char *q11,
	unsigned char *q20, unsigned char *q21,
	unsigned char *q30, unsigned char *q31,
	unsigned int beta, unsigned int tc) {
	int c = tc;
	// δc = (((p0 − q0) << 2) + p1 − q1 + 4) >> 3
	int _p00 = *p00, _p01 = *p01,
		_p10 = *p10, _p11 = *p11,
		_p20 = *p20, _p21 = *p21,
		_p30 = *p30, _p31 = *p31,
		_q00 = *q00, _q01 = *q01,
		_q10 = *q10, _q11 = *q11,
		_q20 = *q20, _q21 = *q21,
		_q30 = *q30, _q31 = *q31;

	int delta0_p = (((_p00 - _q00) << 2) + _p01 - _q01 + 4) >> 3;
	int delta1_p = (((_p10 - _q10) << 2) + _p11 - _q11 + 4) >> 3;
	int delta2_p = (((_p20 - _q20) << 2) + _p21 - _q21 + 4) >> 3;
	int delta3_p = (((_p30 - _q30) << 2) + _p31 - _q31 + 4) >> 3;

	int delta0_q = (((_q00 - _p00) << 2) + _q01 - _p01 + 4) >> 3;
	int delta1_q = (((_q10 - _p10) << 2) + _q11 - _p11 + 4) >> 3;
	int delta2_q = (((_q20 - _p20) << 2) + _q21 - _p21 + 4) >> 3;
	int delta3_q = (((_q30 - _p30) << 2) + _q31 - _p31 + 4) >> 3;

	// clip1
	// clip1
	int Delta0_p = Clip1(delta0_p, c);
	int Delta1_p = Clip1(delta1_p, c);
	int Delta2_p = Clip1(delta2_p, c);
	int Delta3_p = Clip1(delta3_p, c);

	int Delta0_q = Clip1(delta0_q, c);
	int Delta1_q = Clip1(delta1_q, c);
	int Delta2_q = Clip1(delta2_q, c);
	int Delta3_q = Clip1(delta3_q, c);

	int max_v = (1 << 8) - 1;
	*p00 = Clip2(_p00 + Delta0_p, max_v);
	*q00 = Clip2(_q00 - Delta0_q, max_v);

	*p10 = Clip2(_p10 + Delta1_p, max_v);
	*q10 = Clip2(_q10 - Delta1_q, max_v);

	*p20 = Clip2(_p20 + Delta2_p, max_v);
	*q20 = Clip2(_q20 - Delta2_q, max_v);

	*p30 = Clip2(_p30 + Delta3_p, max_v);
	*q30 = Clip2(_q30 - Delta3_q, max_v);

	return;
}

__global__ void DeblockingFilterLumaKernel(unsigned char *gpu_Y_ptr, unsigned char *gpu_vert_bs, unsigned char *gpu_hor_bs,
	unsigned int sample_block_size, unsigned int _new_width, unsigned int _Qp, unsigned num_blocks_x, unsigned int num_blocks_y) {
	unsigned int beta = GetBeta(_Qp);
	unsigned int tc = GetTc(_Qp);

	int block_ind_x = threadIdx.x + blockIdx.x * blockDim.x;
	int block_ind_y = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int half_sample_block_size = sample_block_size / 2;

	unsigned char *Y_block_ptr = gpu_Y_ptr + block_ind_y * sample_block_size * _new_width + block_ind_x * sample_block_size;
	unsigned int _width = _new_width - sample_block_size;

	// compute boundary stregth for upper vertical boundary
	unsigned char BS_ver1 = 0;
	if (block_ind_y > 0) {
		unsigned int tmp_ind = (block_ind_y - 1) * (_width / sample_block_size + 1) + block_ind_x;
		BS_ver1 = gpu_vert_bs[tmp_ind];
	}
	if (BS_ver1 > 0) {
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		unsigned char *p00 = Y_block_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			0 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

		unsigned char *p10 = Y_block_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			1 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

		unsigned char *p20 = Y_block_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = Y_block_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = p00 + 1; unsigned char *q01 = q00 + 1;
		unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = p10 + 1; unsigned char *q11 = q10 + 1;
		unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		unsigned char *q20 = p20 + 1; unsigned char *q21 = q20 + 1;
		unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

		unsigned char *q30 = p30 + 1; unsigned char *q31 = q30 + 1;
		unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

		DeblockingFilterLuma(
			p00, p01, p02, p03,
			p10, p11, p12, p13,
			p20, p21, p22, p23,
			p30, p31, p32, p33,

			q00, q01, q02, q03,
			q10, q11, q12, q13,
			q20, q21, q22, q23,
			q30, q31, q32, q33,
			beta, tc, _Qp);
	}

	// compute boundary stregth for lower vertical boundary
	unsigned char BS_ver2 = 0;
	if (block_ind_y < (num_blocks_y - 1)) {
		unsigned int tmp_ind = block_ind_y * (_width / sample_block_size + 1) + block_ind_x;
		BS_ver2 = gpu_vert_bs[tmp_ind];
	}
	if (BS_ver2 > 0) {
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		unsigned char *p00 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

		unsigned char *p10 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

		unsigned char *p20 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 3) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = p00 + 1; unsigned char *q01 = q00 + 1;
		unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = p10 + 1; unsigned char *q11 = q10 + 1;
		unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		unsigned char *q20 = p20 + 1; unsigned char *q21 = q20 + 1;
		unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

		unsigned char *q30 = p30 + 1; unsigned char *q31 = q30 + 1;
		unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

		DeblockingFilterLuma(
			p00, p01, p02, p03,
			p10, p11, p12, p13,
			p20, p21, p22, p23,
			p30, p31, p32, p33,

			q00, q01, q02, q03,
			q10, q11, q12, q13,
			q20, q21, q22, q23,
			q30, q31, q32, q33,
			beta, tc, _Qp);
	}

	// compute boundary stregth for left horizontal boundary
	unsigned char BS_hor1 = 0;
	if (block_ind_x > 0) {
		unsigned int tmp_ind = block_ind_y * (_width / sample_block_size) + (block_ind_x - 1);
		BS_hor1 = gpu_hor_bs[tmp_ind];
	}
	if (BS_hor1 > 0) {
		// p03, p02, p01, p00
		// p13, p12, p11, p10
		// p23, p22, p21, p00
		// p33, p32, p31, p30
		// ------------------
		// q00, q01, q02, q03
		// q10, q11, q12, q13
		// q20, q21, q22, q23
		// q30, q31, q32, q33
		unsigned char *p00 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			0 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

		unsigned char *p10 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			1 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

		unsigned char *p20 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		unsigned char *q20 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

		unsigned char *q30 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 3) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

		DeblockingFilterLuma(
			p33, p23, p13, p03,
			p32, p22, p12, p02,
			p31, p21, p11, p01,
			p30, p20, p10, p00,

			q00, q10, q20, q30,
			q01, q11, q21, q31,
			q02, q12, q22, q32,
			q03, q13, q23, q33,
			beta, tc, _Qp);
	}

	// compute boundary stregth for right horizontal boundary
	unsigned char BS_hor2 = 0;
	if (block_ind_x < (num_blocks_x - 1)) {
		unsigned int tmp_ind = block_ind_y * (_width / sample_block_size) + block_ind_x;
		BS_hor2 = gpu_hor_bs[tmp_ind];
	}
	if (BS_hor2 > 0) {
		// p03, p02, p01, p00
		// p13, p12, p11, p10
		// p23, p22, p21, p00
		// p33, p32, p31, p30
		// ------------------
		// q00, q01, q02, q03
		// q10, q11, q12, q13
		// q20, q21, q22, q23
		// q30, q31, q32, q33
		unsigned char *p00 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			0 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

		unsigned char *p10 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			1 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

		unsigned char *p20 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		unsigned char *q20 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

		unsigned char *q30 = gpu_Y_ptr +
			num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 3) * num_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

		DeblockingFilterLuma(
			p33, p23, p13, p03,
			p32, p22, p12, p02,
			p31, p21, p11, p01,
			p30, p20, p10, p00,

			q00, q10, q20, q30,
			q01, q11, q21, q31,
			q02, q12, q22, q32,
			q03, q13, q23, q33,
			beta, tc, _Qp);
	}
}

__global__ void DeblockingFilterChromaKernel(unsigned char *gpu_U_ptr,
	unsigned char *gpu_chroma_vert_bs, unsigned char *gpu_chroma_hor_bs,
	unsigned int sample_block_size, unsigned int _new_chroma_width,
	unsigned int _Qp, unsigned num_chroma_blocks_x, unsigned int num_chroma_blocks_y) {
	unsigned int beta = GetBeta(_Qp);
	unsigned int tc = GetTc(_Qp);

	int block_ind_x = threadIdx.x + blockIdx.x * blockDim.x;
	int block_ind_y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int half_sample_block_size = sample_block_size / 2;

	unsigned int _chroma_width = _new_chroma_width - sample_block_size;

	// compute boundary stregth for upper vertical boundary
	unsigned char BS_ver1 = 0;
	if (block_ind_y > 0) {
		unsigned int tmp_ind = (block_ind_y - 1) * (_chroma_width / sample_block_size + 1) + block_ind_x;
		BS_ver1 = gpu_chroma_vert_bs[tmp_ind];

	}
	if (BS_ver1 == 2) {
		// p01, p00 | q00, q01,
		// p11, p10 | q10, q11,
		// p21, p20 | q20, q21,
		// p31, p30 | q30, q31,
		unsigned char *p00 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			0 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

		unsigned char *p10 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			1 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

		unsigned char *p20 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

		unsigned char *p30 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p31 = p30 - 1;

		unsigned char *q00 = p00 + 1; unsigned char *q01 = q00 + 1;

		unsigned char *q10 = p10 + 1; unsigned char *q11 = q10 + 1;

		unsigned char *q20 = p20 + 1; unsigned char *q21 = q20 + 1;

		unsigned char *q30 = p30 + 1; unsigned char *q31 = q30 + 1;

		DeblockingFilterChroma(
			p00, p01,
			p10, p11,
			p20, p21,
			p30, p31,

			q00, q01,
			q10, q11,
			q20, q21,
			q30, q31,
			beta, tc);
	}

	// compute boundary stregth for lower vertical boundary
	unsigned char BS_ver2 = 0;
	if (block_ind_y < (num_chroma_blocks_y - 1)) {
		unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size + 1) + block_ind_x;
		BS_ver2 = gpu_chroma_vert_bs[tmp_ind];
	}
	if (BS_ver2 == 2) {
		// p01, p00 | q00, q01,
		// p11, p10 | q10, q11,
		// p21, p00 | q20, q21,
		// p31, p30 | q30, q31,
		unsigned char *p00 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

		unsigned char *p10 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

		unsigned char *p20 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

		unsigned char *p30 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 3) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1); unsigned char *p31 = p30 - 1;

		unsigned char *q00 = p00 + 1; unsigned char *q01 = q00 + 1;
		unsigned char *q10 = p10 + 1; unsigned char *q11 = q10 + 1;
		unsigned char *q20 = p20 + 1; unsigned char *q21 = q20 + 1;
		unsigned char *q30 = p30 + 1; unsigned char *q31 = q30 + 1;

		DeblockingFilterChroma(
			p00, p01,
			p10, p11,
			p20, p21,
			p30, p31,

			q00, q01,
			q10, q11,
			q20, q21,
			q30, q31,
			beta, tc);
	}

	// compute boundary stregth for left horizontal boundary
	unsigned char BS_hor1 = 0;
	if (block_ind_x > 0) {
		unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size) + (block_ind_x - 1);
		BS_hor1 = gpu_chroma_hor_bs[tmp_ind];
	}
	if (BS_hor1 == 2) {
		// p03, p02, p01, p00
		// p13, p12, p11, p10
		// p23, p22, p21, p00
		// p33, p32, p31, p30
		// ------------------
		// q00, q01, q02, q03
		// q10, q11, q12, q13
		// q20, q21, q22, q23
		// q30, q31, q32, q33
		unsigned char *p20 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(half_sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		DeblockingFilterChroma(
			p33, p23,
			p32, p22,
			p31, p21,
			p30, p20,

			q00, q10,
			q01, q11,
			q02, q12,
			q03, q13,
			beta, tc);
	}

	// compute boundary stregth for right horizontal boundary
	unsigned char BS_hor2 = 0;
	if (block_ind_x < (num_chroma_blocks_x - 1)) {
		unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size) + block_ind_x;
		BS_hor2 = gpu_chroma_hor_bs[tmp_ind];
	}
	if (BS_hor2 == 2) {
		// p03, p02, p01, p00
		// p13, p12, p11, p10
		// p23, p22, p21, p00
		// p33, p32, p31, p30
		// ------------------
		// q00, q01, q02, q03
		// q10, q11, q12, q13
		// q20, q21, q22, q23
		// q30, q31, q32, q33
		unsigned char *p20 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			2 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

		unsigned char *p30 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			3 * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x +
			(sample_block_size - 1);
		unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

		unsigned char *q00 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

		unsigned char *q10 = gpu_U_ptr +
			num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
			(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
			sample_block_size * block_ind_x;
		unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

		DeblockingFilterChroma(
			p33, p23,
			p32, p22,
			p31, p21,
			p30, p20,

			q00, q10,
			q01, q11,
			q02, q12,
			q03, q13,
			beta, tc);
	}

}

void Initialize(char const *file_name, unsigned int width, unsigned int height, unsigned int Qp = 20) {
	// set quantization parameter
	_Qp = Qp;

	// read YUV components
	std::ifstream ifs(file_name, std::ios::binary | std::ios::ate);
	unsigned int length = (unsigned int)ifs.tellg();

	if (length != 3 * width * height / 2) {
		throw "Incorrect file size";
	}
	if (width % sample_block_size != 0 || height % sample_block_size != 0) {
		throw "Width and height of image must be multiplier of sample block size";
	}

	unsigned int half_sample_block_size = sample_block_size / 2;

	_width = width;
	_height = height;

	_new_width = _width + sample_block_size;
	_new_height = _height + sample_block_size;

	_chroma_width = _width / 2;
	_chroma_height = _height / 2;
	_new_chroma_width = _chroma_width + sample_block_size;
	_new_chroma_height = _chroma_height + sample_block_size;

	// allocate pinned memory on host
	cudaError_t status = cudaMallocHost((void **)&Y_pinned_ptr, _new_width * _new_height);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 1\n");
	}
	status = cudaMallocHost((void **)&U_pinned_ptr, _new_chroma_width * _new_chroma_height);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 2\n");
	}
	status = cudaMallocHost((void **)&V_pinned_ptr, _new_chroma_width * _new_chroma_height);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 3\n");
	}

	// read luma component
	ifs.seekg(0, ios::beg);
	for (unsigned int row = 0; row < _height; row++) {
		unsigned char *dst_ptr = Y_pinned_ptr + half_sample_block_size * _new_width + row * _new_width + half_sample_block_size;
		ifs.read((char *)dst_ptr, _width);
	}
	// read chroma components
	for (unsigned int row = 0; row < _chroma_height; row++) {
		unsigned char *dst_ptr = U_pinned_ptr + half_sample_block_size * _new_chroma_width
			+ row * _new_chroma_width + half_sample_block_size;
		ifs.read((char *)dst_ptr, _chroma_width);
	}
	for (unsigned int row = 0; row < _chroma_height; row++) {
		unsigned char *dst_ptr = V_pinned_ptr + half_sample_block_size * _new_chroma_width
			+ row * _new_chroma_width + half_sample_block_size;
		ifs.read((char *)dst_ptr, _chroma_width);
	}
	ifs.close();

	// allocate memory for vertical and horizontal boundaries
	_num_vert_bs = (_width / sample_block_size + 1) * _height / sample_block_size;
	_num_hor_bs = (_height / sample_block_size + 1) * _width / sample_block_size;
	status = cudaMallocHost((void **)&_vert_bs, _num_vert_bs);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 4\n");
	}
	status = cudaMallocHost((void **)&_hor_bs, _num_hor_bs);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 5\n");
	}

	// initialize BS (Boundary Strength) to 2 assuming all blocks are Intra by default
	for (unsigned int i = 0; i < _num_vert_bs; i++) {
		_vert_bs[i] = 2;
		if (i % (_width / sample_block_size + 1) == 0) _vert_bs[i] = 0;
	}
	for (unsigned int i = 0; i < _num_hor_bs; i++) {
		_hor_bs[i] = 2;
		if (i % (_height / sample_block_size + 1) == 0) _hor_bs[i] = 0;
	}

	// allocate memory for vertical and horizontal boundaries
	unsigned int _chroma_width = _width / 2;
	unsigned int _chroma_height = _height / 2;
	_num_chroma_vert_bs = (_chroma_width / sample_block_size + 1) * _chroma_height / sample_block_size;
	_num_chroma_hor_bs = (_chroma_height / sample_block_size + 1) * _chroma_width / sample_block_size;
	status = cudaMallocHost((void **)&_chroma_vert_bs, _num_chroma_vert_bs);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 5\n");
	}
	status = cudaMallocHost((void **)&_chroma_hor_bs, _num_chroma_hor_bs);
	if (status != cudaSuccess) {
		printf("Error returned from pinned host memory allocation - 5\n");
	}

	// initialize boundary strength for chromas
	for (unsigned int i = 0; i < _num_chroma_vert_bs; i++) {
		_chroma_vert_bs[i] = 2;
		if (i % (_chroma_width / sample_block_size + 1) == 0) _chroma_vert_bs[i] = 0;
	}
	for (unsigned int i = 0; i < _num_chroma_hor_bs; i++) {
		_chroma_hor_bs[i] = 2;
		if (i % (_chroma_height / sample_block_size + 1) == 0) _chroma_hor_bs[i] = 0;
	}
}

void Release() {
	// release GPU buffers for Y, U, V
	cudaFree(_gpu_Y_ptr);
	cudaFree(_gpu_U_ptr);
	cudaFree(_gpu_V_ptr);

	// release GPU buffers for boundary stregths
	cudaFree(_gpu_vert_bs);
	cudaFree(_gpu_hor_bs);
	cudaFree(_gpu_chroma_vert_bs);
	cudaFree(_gpu_chroma_hor_bs);

	// release Host pinned memory
	cudaFreeHost(Y_pinned_ptr);
	cudaFreeHost(U_pinned_ptr);
	cudaFreeHost(V_pinned_ptr);

	cudaFreeHost(_vert_bs);
	cudaFreeHost(_hor_bs);
	cudaFreeHost(_chroma_vert_bs);
	cudaFreeHost(_chroma_hor_bs);
}

void Save(char const *output_file_name) {
	unsigned int half_sample_block_size = sample_block_size / 2;

	std::ofstream output_file(output_file_name, ios::out | ios::binary);
	// copy filtered luma back
	for (unsigned int row = 0; row < _height; row++) {
		unsigned char *src_ptr = Y_pinned_ptr + half_sample_block_size * _new_width +
			row * _new_width + half_sample_block_size;
		output_file.write((char const *)src_ptr, _width);
	}

	// copy filtered chroma components back
	for (unsigned int row = 0; row < _chroma_height; row++) {
		unsigned char *src_ptr = U_pinned_ptr + half_sample_block_size * _new_chroma_width
			+ row * _new_chroma_width + half_sample_block_size;
		output_file.write((char const *)src_ptr, _chroma_width);
	}
	for (unsigned int row = 0; row < _chroma_height; row++) {
		unsigned char *src_ptr = V_pinned_ptr + half_sample_block_size * _new_chroma_width
			+ row * _new_chroma_width + half_sample_block_size;
		output_file.write((char const *)src_ptr, _chroma_width);
	}
	output_file.close();
}

void ExecuteGpu(std::string const &input_file_name, std::string const &output_file_name,
	unsigned int width, unsigned int height, unsigned int Qp,
	unsigned dimx1, unsigned int dimy1, unsigned dimx2, unsigned int dimy2) {
	Initialize(input_file_name.c_str(), width, height, Qp);

	// allocate GPU global memory for Y, U, V
	cudaMalloc((unsigned char **)&_gpu_Y_ptr, _new_width * _new_height);
	cudaMalloc((unsigned char **)&_gpu_U_ptr, _new_chroma_width * _new_chroma_height);
	cudaMalloc((unsigned char **)&_gpu_V_ptr, _new_chroma_width * _new_chroma_height);

	// allocate GPU global memory for boundary stregths
	cudaMalloc((unsigned char **)&_gpu_vert_bs, _num_vert_bs);
	cudaMalloc((unsigned char **)&_gpu_hor_bs, _num_hor_bs);
	cudaMalloc((unsigned char **)&_gpu_chroma_vert_bs, _num_chroma_vert_bs);
	cudaMalloc((unsigned char **)&_gpu_chroma_hor_bs, _num_chroma_hor_bs);

	auto start = std::chrono::system_clock::now();
	// transfer data from host to device
	cudaMemcpy(_gpu_Y_ptr, Y_pinned_ptr, _new_width * _new_height, cudaMemcpyHostToDevice);
	cudaMemcpy(_gpu_U_ptr, U_pinned_ptr, _new_chroma_width * _new_chroma_height, cudaMemcpyHostToDevice);
	cudaMemcpy(_gpu_V_ptr, V_pinned_ptr, _new_chroma_width * _new_chroma_height, cudaMemcpyHostToDevice);

	// transfer data from host to device
	cudaMemcpy(_gpu_vert_bs, _vert_bs, _num_vert_bs, cudaMemcpyHostToDevice);
	cudaMemcpy(_gpu_hor_bs, _hor_bs, _num_hor_bs, cudaMemcpyHostToDevice);
	cudaMemcpy(_gpu_chroma_vert_bs, _chroma_vert_bs, _num_chroma_vert_bs, cudaMemcpyHostToDevice);
	cudaMemcpy(_gpu_chroma_hor_bs, _chroma_hor_bs, _num_chroma_hor_bs, cudaMemcpyHostToDevice);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> gpu_buffers_copy_operation_time = end - start;

	unsigned int num_blocks_x = _new_width / sample_block_size;
	unsigned int num_blocks_y = _new_height / sample_block_size;

	unsigned int num_chroma_blocks_x = _new_chroma_width / sample_block_size;
	unsigned int num_chroma_blocks_y = _new_chroma_height / sample_block_size;

	start = std::chrono::system_clock::now();
	dim3 block(dimx1, dimy1);
	dim3 grid((num_blocks_x + block.x - 1) / block.x, (num_blocks_y + block.y - 1) / block.y);
	DeblockingFilterLumaKernel << <grid, block >> >(_gpu_Y_ptr, _gpu_vert_bs, _gpu_hor_bs, sample_block_size,
		_new_width, Qp, num_blocks_x, num_blocks_y);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error - 1: %s\n", cudaGetErrorString(err));
	dim3 chroma_block(dimx2, dimy2);
	dim3 chroma_grid((num_chroma_blocks_x + chroma_block.x - 1) / chroma_block.x,
		(num_chroma_blocks_y + chroma_block.y - 1) / chroma_block.y);
	DeblockingFilterChromaKernel << <chroma_grid, chroma_block >> >(_gpu_U_ptr,
		_gpu_chroma_vert_bs, _gpu_chroma_hor_bs, sample_block_size,
		_new_chroma_width, _Qp, num_chroma_blocks_x, num_chroma_blocks_y);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error - 2: %s\n", cudaGetErrorString(err));
	DeblockingFilterChromaKernel << <chroma_grid, chroma_block >> >(_gpu_V_ptr,
		_gpu_chroma_vert_bs, _gpu_chroma_hor_bs, sample_block_size,
		_new_chroma_width, _Qp, num_chroma_blocks_x, num_chroma_blocks_y);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error - 3: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> gpu_execution_time = end - start;
	std::cout << "Execution Time without copy on GPU: " << gpu_execution_time.count() << "s" << std::endl;

	start = std::chrono::system_clock::now();
	// copy kernel result back to host side
	cudaMemcpy(Y_pinned_ptr, _gpu_Y_ptr, _new_width * _new_height, cudaMemcpyDeviceToHost);
	cudaMemcpy(U_pinned_ptr, _gpu_U_ptr, _new_chroma_width * _new_chroma_height, cudaMemcpyDeviceToHost);
	cudaMemcpy(V_pinned_ptr, _gpu_V_ptr, _new_chroma_width * _new_chroma_height, cudaMemcpyDeviceToHost);
	end = std::chrono::system_clock::now();
	gpu_buffers_copy_operation_time += end - start;

	std::cout << "Execution Time with copy on GPU: " << gpu_execution_time.count() + gpu_buffers_copy_operation_time.count() << "s" << std::endl;
	std::cout << "Copy Operation Time with GPU buffers: " << gpu_buffers_copy_operation_time.count() << "s" << std::endl;
	Save(output_file_name.c_str());
	Release();
}
