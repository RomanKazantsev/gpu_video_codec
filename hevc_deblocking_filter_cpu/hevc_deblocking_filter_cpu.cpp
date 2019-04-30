﻿/*
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

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>

using namespace std;

// Read YV12 format frame and deblock
class ReadYuvFrame {
public:
	ReadYuvFrame(char const *file_name, unsigned int width, unsigned int height, unsigned int Qp = 20) {
		// set quantization parameter
		_Qp = Qp;

		// read YUV components
		std::ifstream ifs(file_name, std::ios::binary | std::ios::ate);
		unsigned int length = (unsigned int) ifs.tellg();

		if (length != 3 * width * height / 2) {
			throw "Incorrect file size";
		}
		if (width % sample_block_size != 0 || height % sample_block_size != 0) {
			throw "Width and height of image must be multiplier of sample block size";
		}

		_width = width;
		_height = height;

		_Y.reset(new unsigned char[width * height]);
		_U.reset(new unsigned char[width * height / 4]);
		_V.reset(new unsigned char[width * height / 4]);

		ifs.seekg(0, ios::beg);
		ifs.read((char *)_Y.get(), width * height);
		ifs.read((char *)_U.get(), width * height / 4);
		ifs.read((char *)_V.get(), width * height / 4);

		ifs.close();

		// allocate memory for vertical and horizontal boundaries
		_num_vert_bs = (_width / sample_block_size + 1) * _height / sample_block_size;
		_num_hor_bs = (_height / sample_block_size + 1) * _width / sample_block_size;
		_vert_bs.reset(new unsigned char[_num_vert_bs]);
		_hor_bs.reset(new unsigned char[_num_hor_bs]);

		// initialize BS (Boundary Strength) to 2 assuming all blocks are Intra by default
		for (unsigned int i = 0; i < _num_vert_bs; i++) {
			_vert_bs.get()[i] = 2;
			if (i % (_width / sample_block_size + 1) == 0) _vert_bs.get()[i] = 0;
		}
		for (unsigned int i = 0; i < _num_hor_bs; i++) {
			_hor_bs.get()[i] = 2;
			if (i % (_height / sample_block_size + 1) == 0) _hor_bs.get()[i] = 0;
		}

		// allocate memory for vertical and horizontal boundaries
		unsigned int _chroma_width = _width / 2;
		unsigned int _chroma_height = _height / 2;
		_num_chroma_vert_bs = (_chroma_width / sample_block_size + 1) * _chroma_height / sample_block_size;
		_num_chroma_hor_bs = (_chroma_height / sample_block_size + 1) * _chroma_width / sample_block_size;
		_chroma_vert_bs.reset(new unsigned char[_num_chroma_vert_bs]);
		_chroma_hor_bs.reset(new unsigned char[_num_chroma_hor_bs]);

		// initialize boundary strength for chromas
		for (unsigned int i = 0; i < _num_chroma_vert_bs; i++) {
			_chroma_vert_bs.get()[i] = 2;
			if (i % (_chroma_width / sample_block_size + 1) == 0) _chroma_vert_bs.get()[i] = 0;
		}
		for (unsigned int i = 0; i < _num_chroma_hor_bs; i++) {
			_chroma_hor_bs.get()[i] = 2;
			if (i % (_chroma_height / sample_block_size + 1) == 0) _chroma_hor_bs.get()[i] = 0;
		}
	}

	void SetBoundaryStrenght(unsigned char *vert_bs, unsigned int num_vert_bs,
		unsigned char *hor_bs, unsigned int num_hor_bs) {
		if (_num_hor_bs != num_hor_bs || _num_vert_bs != num_vert_bs)
			throw "Incorrect size of input boundary strenght array";
		// set BS (Boundary Strength)
		for (unsigned int i = 0; i < _num_vert_bs; i++) {
			_vert_bs.get()[i] = vert_bs[i];
		}
		for (unsigned int i = 0; i < _num_hor_bs; i++) {
			_hor_bs.get()[i] = hor_bs[i];
		}
		return;
	}

	void DeblockingFilter() {
		unsigned int beta = GetBeta(_Qp);
		unsigned int tc = GetTc(_Qp);

		unsigned int half_sample_block_size = sample_block_size / 2;

		// create auxiliary memory
		unsigned int new_width = _width + sample_block_size;
		unsigned int new_height = _height + sample_block_size;
		unsigned int _chroma_width = _width / 2;
		unsigned int _chroma_height = _height / 2;
		unsigned int new_chroma_width = _chroma_width + sample_block_size;
		unsigned int new_chroma_height = _chroma_height + sample_block_size;
		unique_ptr<unsigned char> ext_Y;
		ext_Y.reset(new unsigned char[new_width * new_height]);
		unique_ptr<unsigned char> ext_U;
		ext_U.reset(new unsigned char[new_chroma_width * new_chroma_height]);
		unique_ptr<unsigned char> ext_V;
		ext_V.reset(new unsigned char[new_chroma_width * new_chroma_height]);

		// copy to auxiliary memory for luma
		for (unsigned int row = 0; row < _height; row++) {
			unsigned char *dst_ptr = ext_Y.get() + half_sample_block_size * new_width + row * new_width + half_sample_block_size;
			unsigned char *src_ptr = _Y.get() + row * _width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _width);
		}

		unsigned int num_blocks_x = new_width / sample_block_size;
		unsigned int num_blocks_y = new_height / sample_block_size;

		// filter luma
		for (unsigned int block_ind_x = 0; block_ind_x < num_blocks_x; block_ind_x++) {
			for (unsigned int block_ind_y = 0; block_ind_y < num_blocks_y; block_ind_y++) {
				unsigned char *Y_block_ptr = ext_Y.get() +
					block_ind_y * sample_block_size * new_width +
					block_ind_x * sample_block_size;
				unsigned char *U_block_ptr = ext_U.get() +
					block_ind_y * half_sample_block_size * new_width / 2 +
					block_ind_x * half_sample_block_size;
				unsigned char *V_block_ptr = ext_V.get() +
					block_ind_y * half_sample_block_size * new_width / 2 +
					block_ind_x * half_sample_block_size;

				// compute boundary stregth for upper vertical boundary
				unsigned char BS_ver1 = 0;
				if (block_ind_y > 0) {
					unsigned int tmp_ind = (block_ind_y - 1) * (_width / sample_block_size + 1) + block_ind_x;
					BS_ver1 = _vert_bs.get()[tmp_ind];
				}
				if (BS_ver1 > 0) {
					// p03, p02, p01, p00 | q00, q01, q02, q03
					// p13, p12, p11, p10 | q10, q11, q12, q13
					// p23, p22, p21, p00 | q20, q21, q22, q23
					// p33, p32, p31, p30 | q30, q31, q32, q33
					unsigned char *p00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_Y.get() +
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
						beta, tc);
				}

				// compute boundary stregth for lower vertical boundary
				unsigned char BS_ver2 = 0;
				if (block_ind_y < (num_blocks_y - 1)) {
					unsigned int tmp_ind = block_ind_y * (_width / sample_block_size + 1) + block_ind_x;
					BS_ver2 = _vert_bs.get()[tmp_ind];
				}
				if (BS_ver2 > 0) {
					// p03, p02, p01, p00 | q00, q01, q02, q03
					// p13, p12, p11, p10 | q10, q11, q12, q13
					// p23, p22, p21, p00 | q20, q21, q22, q23
					// p33, p32, p31, p30 | q30, q31, q32, q33
					unsigned char *p00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_Y.get() +
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
						beta, tc);
				}

				// compute boundary stregth for left horizontal boundary
				unsigned char BS_hor1 = 0;
				if (block_ind_x > 0) {
					unsigned int tmp_ind = block_ind_y * (_width / sample_block_size) + (block_ind_x - 1);
					BS_hor1 = _hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_Y.get() +
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
						beta, tc);
				}

				// compute boundary stregth for right horizontal boundary
				unsigned char BS_hor2 = 0;
				if (block_ind_x < (num_blocks_x - 1)) {
					unsigned int tmp_ind = block_ind_y * (_width / sample_block_size) + block_ind_x;
					BS_hor2 = _hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_Y.get() +
						num_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_Y.get() +
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
						beta, tc);
				}
			}
		}

		// copy filtered luma back
		for (unsigned int row = 0; row < _height; row++) {
			unsigned char *src_ptr = ext_Y.get() + half_sample_block_size * new_width + row * new_width + half_sample_block_size;
			unsigned char *dst_ptr = _Y.get() + row * _width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _width);
		}

		// copy to auxiliary memory for chroma components
		for (unsigned int row = 0; row < _chroma_height; row++) {
			unsigned char *dst_ptr = ext_U.get() + half_sample_block_size * new_chroma_width
				+ row * new_chroma_width + half_sample_block_size;
			unsigned char *src_ptr = _U.get() + row * _chroma_width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _chroma_width);
		}
		for (unsigned int row = 0; row < _chroma_height; row++) {
			unsigned char *dst_ptr = ext_V.get() + half_sample_block_size * new_chroma_width
				+ row * new_chroma_width + half_sample_block_size;
			unsigned char *src_ptr = _V.get() + row * _chroma_width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _chroma_width);
		}

		unsigned int num_chroma_blocks_x = new_chroma_width / sample_block_size;
		unsigned int num_chroma_blocks_y = new_chroma_height / sample_block_size;
		// filter chroma component U
		for (unsigned int block_ind_x = 0; block_ind_x < num_chroma_blocks_x; block_ind_x++) {
			for (unsigned int block_ind_y = 0; block_ind_y < num_chroma_blocks_y; block_ind_y++) {
				// compute boundary stregth for upper vertical boundary
				unsigned char BS_ver1 = 0;
				if (block_ind_y > 0) {
					unsigned int tmp_ind = (block_ind_y - 1) * (_chroma_width / sample_block_size + 1) + block_ind_x;
					BS_ver1 = _chroma_vert_bs.get()[tmp_ind];
					
				}
				if (BS_ver1 == 2) {
					// p01, p00 | q00, q01,
					// p11, p10 | q10, q11,
					// p21, p20 | q20, q21,
					// p31, p30 | q30, q31,
					unsigned char *p00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

					unsigned char *p10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

					unsigned char *p20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

					unsigned char *p30 = ext_U.get() +
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
				if (block_ind_y < (num_blocks_y - 1)) {
					unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size + 1) + block_ind_x;
					BS_ver2 = _chroma_vert_bs.get()[tmp_ind];
				}
				if (BS_ver2 == 2) {
					// p01, p00 | q00, q01,
					// p11, p10 | q10, q11,
					// p21, p00 | q20, q21,
					// p31, p30 | q30, q31,
					unsigned char *p00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

					unsigned char *p10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

					unsigned char *p20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

					unsigned char *p30 = ext_U.get() +
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
					BS_hor1 = _chroma_hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 3) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

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
				if (block_ind_x < (num_blocks_x - 1)) {
					unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size) + block_ind_x;
					BS_hor2 = _chroma_hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_U.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 3) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

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
		}

		// filter chroma component V
		for (unsigned int block_ind_x = 0; block_ind_x < num_chroma_blocks_x; block_ind_x++) {
			for (unsigned int block_ind_y = 0; block_ind_y < num_chroma_blocks_y; block_ind_y++) {
				// compute boundary stregth for upper vertical boundary
				unsigned char BS_ver1 = 0;
				if (block_ind_y > 0) {
					unsigned int tmp_ind = (block_ind_y - 1) * (_chroma_width / sample_block_size + 1) + block_ind_x;
					BS_ver1 = _chroma_vert_bs.get()[tmp_ind];

				}
				if (BS_ver1 == 2) {
					// p01, p00 | q00, q01,
					// p11, p10 | q10, q11,
					// p21, p20 | q20, q21,
					// p31, p30 | q30, q31,
					unsigned char *p00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

					unsigned char *p10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

					unsigned char *p20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

					unsigned char *p30 = ext_V.get() +
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
				if (block_ind_y < (num_blocks_y - 1)) {
					unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size + 1) + block_ind_x;
					BS_ver2 = _chroma_vert_bs.get()[tmp_ind];
				}
				if (BS_ver2 == 2) {
					// p01, p00 | q00, q01,
					// p11, p10 | q10, q11,
					// p21, p00 | q20, q21,
					// p31, p30 | q30, q31,
					unsigned char *p00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p01 = p00 - 1;

					unsigned char *p10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p11 = p10 - 1;

					unsigned char *p20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1); unsigned char *p21 = p20 - 1;

					unsigned char *p30 = ext_V.get() +
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
					BS_hor1 = _chroma_hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(half_sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 3) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

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
				if (block_ind_x < (num_blocks_x - 1)) {
					unsigned int tmp_ind = block_ind_y * (_chroma_width / sample_block_size) + block_ind_x;
					BS_hor2 = _chroma_hor_bs.get()[tmp_ind];
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
					unsigned char *p00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						0 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p01 = p00 - 1; unsigned char *p02 = p01 - 1; unsigned char *p03 = p02 - 1;

					unsigned char *p10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						1 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p11 = p10 - 1; unsigned char *p12 = p11 - 1; unsigned char *p13 = p12 - 1;

					unsigned char *p20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						2 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p21 = p20 - 1; unsigned char *p22 = p21 - 1; unsigned char *p23 = p22 - 1;

					unsigned char *p30 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						3 * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x +
						(sample_block_size - 1);
					unsigned char *p31 = p30 - 1; unsigned char *p32 = p31 - 1; unsigned char *p33 = p32 - 1;

					unsigned char *q00 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q01 = q00 + 1; unsigned char *q02 = q01 + 1; unsigned char *q03 = q02 + 1;

					unsigned char *q10 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 1) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q11 = q10 + 1; unsigned char *q12 = q11 + 1; unsigned char *q13 = q12 + 1;

					unsigned char *q20 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 2) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q21 = q20 + 1; unsigned char *q22 = q21 + 1; unsigned char *q23 = q22 + 1;

					unsigned char *q30 = ext_V.get() +
						num_chroma_blocks_x * sample_block_size * sample_block_size * block_ind_y +
						(sample_block_size / 2 + 3) * num_chroma_blocks_x * sample_block_size +
						sample_block_size * block_ind_x;
					unsigned char *q31 = q30 + 1; unsigned char *q32 = q31 + 1; unsigned char *q33 = q32 + 1;

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
		}

		// copy filtered chroma components back
		for (unsigned int row = 0; row < _chroma_height; row++) {
			unsigned char *dst_ptr = ext_U.get() + half_sample_block_size * new_chroma_width
				+ row * new_chroma_width + half_sample_block_size;
			unsigned char *src_ptr = _U.get() + row * _chroma_width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _chroma_width);
		}
		for (unsigned int row = 0; row < _chroma_height; row++) {
			unsigned char *dst_ptr = ext_V.get() + half_sample_block_size * new_chroma_width
				+ row * new_chroma_width + half_sample_block_size;
			unsigned char *src_ptr = _V.get() + row * _chroma_width;
			std::memcpy((void *)dst_ptr, (void *)src_ptr, _chroma_width);
		}
	}

	void Save(char const *output_file_name) {
		unsigned char const *y_ptr = _Y.get();
		unsigned char const *u_ptr = _U.get();
		unsigned char const *v_ptr = _V.get();

		std::ofstream output_file(output_file_name, ios::out | ios::binary);
		output_file.write((char const *)y_ptr, _width * _height);
		output_file.write((char const *)u_ptr, _width * _height / 4);
		output_file.write((char const *)v_ptr, _width * _height / 4);
		output_file.close();
	}

private:
	const int sample_block_size = 8;
	unsigned int _width;
	unsigned int _height;
	unique_ptr<unsigned char> _Y; // width * height - values in range [0..255]
	unique_ptr<unsigned char> _U; // width * height // 4
	unique_ptr<unsigned char> _V; // width * height // 4

	// strenght of vertical and horizontal boundaries for luma
	unsigned int _num_vert_bs;
	unsigned int _num_hor_bs;
	unique_ptr<unsigned char> _vert_bs;
	unique_ptr<unsigned char> _hor_bs;

	// strength of vertical and horizontal boundaries for chroma components
	unsigned int _num_chroma_vert_bs;
	unsigned int _num_chroma_hor_bs;
	unique_ptr<unsigned char> _chroma_vert_bs;
	unique_ptr<unsigned char> _chroma_hor_bs;

	unsigned int _Qp; // quantization parameter

	unsigned int GetBeta(unsigned int QP) const {
		if (QP <= 15) return 0;
		if (QP <= 30) return QP - 10;
		return 2 * QP - 40;
	}

	unsigned int GetTc(unsigned int QP) const {
		if (QP <= 18) return 0;
		if (QP <= 27) return 1;
		if (QP <= 30) return 2;
		if (QP <= 34) return 3;
		if (QP <= 37) return 4;
		if (QP <= 40) return 5;
		if (QP <= 41) return 6;
		return (9 * QP - 285) / 14;
	}

	bool CheckLocalAdaptivity(
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
		if ((std::abs(p02 - 2 * p01 + p00) + std::abs(p32 - 2 * p31 + p30) +
			std::abs(q02 - 2 * q01 + q00) + std::abs(q32 - 2 * q31 + q30)) < beta) return true;
		return false;
	}

	bool IsStrongFilterToUse(
		int p00, int p01, int p02, int p03,
		int p30, int p31, int p32, int p33,
		int q00, int q01, int q02, int q03,
		int q30, int q31, int q32, int q33,
		int beta, int tc) {
		// check condition (2)
		bool cond2 = false;
		if (((std::abs(p02 - 2 * p01 + p00) + std::abs(q02 - 2 * q01 + q00)) < beta / 8) &&
			((std::abs(p32 - 2 * p31 + p30) + std::abs(q32 - 2 * q31 + q30)) < beta / 8)) cond2 = true;

		// check condition (3)
		bool cond3 = false;
		if (((std::abs(p03 - p00) + std::abs(q00 - q03)) < beta / 8) &&
			((std::abs(p33 - p30) + std::abs(q30 - q33)) < beta / 8)) cond3 = true;

		// check condition (4)
		bool cond4 = false;
		if ((std::abs(p00 - q00) < 5 * tc / 2) &&
			(std::abs(p30 - q30) < 5 * tc / 2)) cond4 = true;

		if (cond2 && cond3 && cond4) return true;
		return false;
	}

	bool AreP0P1Modified(
		int p00, int p10, int p20, int p30,
		int p03, int p13, int p23, int p33,
		int beta) {
		if ((std::abs(p20 - 2 * p10 + p00) + std::abs(p23 - 2 * p13 + p03)) < 3 * beta / 16)
			return true;
		return false;
	}

	// clip value so that it will be in a range [-c; c]
	int Clip1(int delta, int c) {
		if (c < 0) throw "c parameter is negative";
		return std::min(std::max(-c, delta), c);
	}

	// clip value so that it will be in a range [0; c]
	int Clip2(int value, int c) {
		if (c <= 0) throw "c parameter is negative";
		return std::min(std::max(0, value), c);
	}

	void ApplyStrongFilter(
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

		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33

		int _p00 = *p00, _p01 = *p01, _p02 = *p02, _p03 = *p03, _q00 = *q00, _q01 = *q01, _q02 = *q02, _q03 = *q03;
		int _p10 = *p10, _p11 = *p11, _p12 = *p12, _p13 = *p13, _q10 = *q10, _q11 = *q11, _q12 = *q12, _q13 = *q13;
		int _p20 = *p20, _p21 = *p21, _p22 = *p22, _p23 = *p23, _q20 = *q20, _q21 = *q21, _q22 = *q22, _q23 = *q23;
		int _p30 = *p30, _p31 = *p31, _p32 = *p32, _p33 = *p33, _q30 = *q30, _q31 = *q31, _q32 = *q32, _q33 = *q33;

		// compute deltas for P block
		// δ0s = (p2 + 2p1 − 6p0 + 2q0 + q1 + 4) >> 3
		int delta00_p = (_p02 + 2 * _p01 - 6 * _p00 + 2 * _q00 + _q01 + 4) >> 3;
		int delta10_p = (_p12 + 2 * _p11 - 6 * _p10 + 2 * _q10 + _q11 + 4) >> 3;
		int delta20_p = (_p22 + 2 * _p21 - 6 * _p20 + 2 * _q20 + _q21 + 4) >> 3;
		int delta30_p = (_p32 + 2 * _p31 - 6 * _p30 + 2 * _q30 + _q31 + 4) >> 3;

		// δ1s = (p2 − 3p1 + p0 + q0 + 2) >> 2
		int delta01_p = (_p02 - 3 * _p01 + _p00 + _q00 + 2) >> 2;
		int delta11_p = (_p12 - 3 * _p11 + _p10 + _q10 + 2) >> 2;
		int delta21_p = (_p22 - 3 * _p21 + _p20 + _q20 + 2) >> 2;
		int delta31_p = (_p32 - 3 * _p31 + _p30 + _q30 + 2) >> 2;

		// δ2s = (2p3 − 5p2 + p1 + p0 + q0 + 4) >> 3
		int delta02_p = (2 * _p03 - 5 * _p02 + _p01 + _p00 + _q00 + 4) >> 3;
		int delta12_p = (2 * _p13 - 5 * _p12 + _p11 + _p10 + _q10 + 4) >> 3;
		int delta22_p = (2 * _p23 - 5 * _p22 + _p21 + _p20 + _q20 + 4) >> 3;
		int delta32_p = (2 * _p33 - 5 * _p32 + _p31 + _p30 + _q30 + 4) >> 3;

		// compute deltas for Q block
		// δ0s = (q2 + 2q1 − 6q0 + 2p0 + p1 + 4) >> 3
		int delta00_q = (_q02 + 2 * _q01 - 6 * _q00 + 2 * _p00 + _p01 + 4) >> 3;
		int delta10_q = (_q12 + 2 * _q11 - 6 * _q10 + 2 * _p10 + _p11 + 4) >> 3;
		int delta20_q = (_q22 + 2 * _q21 - 6 * _q20 + 2 * _p20 + _p21 + 4) >> 3;
		int delta30_q = (_q32 + 2 * _q31 - 6 * _q30 + 2 * _p30 + _p31 + 4) >> 3;

		// δ1s = (q2 − 3q1 + q0 + p0 + 2) >> 2
		int delta01_q = (_q02 - 3 * _q01 + _q00 + _p00 + 2) >> 2;
		int delta11_q = (_q12 - 3 * _q11 + _q10 + _p10 + 2) >> 2;
		int delta21_q = (_q22 - 3 * _q21 + _q20 + _p20 + 2) >> 2;
		int delta31_q = (_q32 - 3 * _q31 + _q30 + _p30 + 2) >> 2;

		// δ2s = (2q3 − 5q2 + q1 + q0 + p0 + 4) >> 3
		int delta02_q = (2 * _q03 - 5 * _q02 + _q01 + _q00 + _p00 + 4) >> 3;
		int delta12_q = (2 * _q13 - 5 * _q12 + _q11 + _q10 + _p10 + 4) >> 3;
		int delta22_q = (2 * _q23 - 5 * _q22 + _q21 + _q20 + _p20 + 4) >> 3;
		int delta32_q = (2 * _q33 - 5 * _q32 + _q31 + _q30 + _p30 + 4) >> 3;

		// clip deltas for P block
		int c = 2 * GetTc(QP);
		delta00_p = Clip1(delta00_p, c); delta10_p = Clip1(delta10_p, c); delta20_p = Clip1(delta20_p, c); delta30_p = Clip1(delta30_p, c);
		delta01_p = Clip1(delta01_p, c); delta11_p = Clip1(delta11_p, c); delta21_p = Clip1(delta21_p, c); delta31_p = Clip1(delta31_p, c);
		delta02_p = Clip1(delta02_p, c); delta12_p = Clip1(delta12_p, c); delta22_p = Clip1(delta22_p, c); delta32_p = Clip1(delta32_p, c);
		
		// clip deltas for Q block
		delta00_q = Clip1(delta00_q, c); delta10_q = Clip1(delta10_q, c); delta20_q = Clip1(delta20_q, c); delta30_q = Clip1(delta30_q, c);
		delta01_q = Clip1(delta01_q, c); delta11_q = Clip1(delta11_q, c); delta21_q = Clip1(delta21_q, c); delta31_q = Clip1(delta31_q, c);
		delta02_q = Clip1(delta02_q, c); delta12_q = Clip1(delta12_q, c); delta22_q = Clip1(delta22_q, c); delta32_q = Clip1(delta32_q, c);

		// filter pixels
		int max_v = 1 << 8;
		// filter and clip values for P block
		*p00 = Clip2(_p00 + delta00_p, max_v); *p10 = Clip2(_p10 + delta10_p, max_v); *p20 = Clip2(_p20 + delta20_p, max_v); *p30 = Clip2(_p30 + delta30_p, max_v);
		*p01 = Clip2(_p01 + delta01_p, max_v); *p11 = Clip2(_p11 + delta11_p, max_v); *p21 = Clip2(_p21 + delta21_p, max_v); *p31 = Clip2(_p31 + delta31_p, max_v);
		*p02 = Clip2(_p02 + delta02_p, max_v); *p12 = Clip2(_p12 + delta12_p, max_v); *p22 = Clip2(_p22 + delta22_p, max_v); *p32 = Clip2(_p32 + delta32_p, max_v);

		// filter and clip values for Q block
		*q00 = Clip2(_q00 - delta00_q, max_v); *q10 = Clip2(_q10 + delta10_q, max_v); *q20 = Clip2(_q20 + delta20_q, max_v); *q30 = Clip2(_q30 + delta30_q, max_v);
		*q01 = Clip2(_q01 - delta01_q, max_v); *q11 = Clip2(_q11 + delta11_q, max_v); *q21 = Clip2(_q21 + delta21_q, max_v); *q31 = Clip2(_q31 + delta31_q, max_v);
		*q02 = Clip2(_q02 - delta02_q, max_v); *q12 = Clip2(_q12 + delta12_q, max_v); *q22 = Clip2(_q22 + delta22_q, max_v); *q32 = Clip2(_q32 + delta32_q, max_v);
		return;
	}

	void ApplyNormalFilter(
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
		int max_v = 1 << 8;
		// p03, p02, p01, p00 | q00, q01, q02, q03
		// p13, p12, p11, p10 | q10, q11, q12, q13
		// p23, p22, p21, p00 | q20, q21, q22, q23
		// p33, p32, p31, p30 | q30, q31, q32, q33
		unsigned int tc = GetTc(QP);
		unsigned int beta = GetBeta(QP);
		int c = 2 * tc;
		int c2 = tc / 2;

		int _p00 = *p00, _p01 = *p01, _p02 = *p02, _p03 = *p03, _q00 = *q00, _q01 = *q01, _q02 = *q02, _q03 = *q03;
		int _p10 = *p10, _p11 = *p11, _p12 = *p12, _p13 = *p13, _q10 = *q10, _q11 = *q11, _q12 = *q12, _q13 = *q13;
		int _p20 = *p20, _p21 = *p21, _p22 = *p22, _p23 = *p23, _q20 = *q20, _q21 = *q21, _q22 = *q22, _q23 = *q23;
		int _p30 = *p30, _p31 = *p31, _p32 = *p32, _p33 = *p33, _q30 = *q30, _q31 = *q31, _q32 = *q32, _q33 = *q33;

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

	void DeblockingFilterLuma(
		unsigned char *p00, unsigned char *p01, unsigned char *p02, unsigned char *p03,
		unsigned char *p10, unsigned char *p11, unsigned char *p12, unsigned char *p13,
		unsigned char *p20, unsigned char *p21, unsigned char *p22, unsigned char *p23,
		unsigned char *p30, unsigned char *p31, unsigned char *p32, unsigned char *p33,

		unsigned char *q00, unsigned char *q01, unsigned char *q02, unsigned char *q03,
		unsigned char *q10, unsigned char *q11, unsigned char *q12, unsigned char *q13,
		unsigned char *q20, unsigned char *q21, unsigned char *q22, unsigned char *q23,
		unsigned char *q30, unsigned char *q31, unsigned char *q32, unsigned char *q33,
		unsigned int beta, unsigned int tc) {
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
				p00, p01, p02, p03,
				p10, p11, p12, p13,
				p20, p21, p22, p23,
				p30, p31, p32, p33,
				q00, q01, q02, q03,
				q10, q11, q12, q13,
				q20, q21, q22, q23,
				q30, q31, q32, q33,
				_Qp);
		}
	}

	void DeblockingFilterChroma(
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

		int delta0 = (((_p00 - _q00) << 2) + _p01 - _q01 + 4) >> 3;
		int delta1 = (((_p10 - _q10) << 2) + _p11 - _q11 + 4) >> 3;
		int delta2 = (((_p20 - _q20) << 2) + _p21 - _q21 + 4) >> 3;
		int delta3 = (((_p30 - _q30) << 2) + _p31 - _q31 + 4) >> 3;

		// clip1
		int Delta0 = Clip1(delta0, c);
		int Delta1 = Clip1(delta1, c);
		int Delta2 = Clip1(delta2, c);
		int Delta3 = Clip1(delta3, c);

		int max_v = 1 << 8;
		*p00 = Clip2(_p00 + Delta0, max_v);
		*q00 = Clip2(_q00 - Delta0, max_v);

		*p10 = Clip2(_p10 + Delta1, max_v);
		*q10 = Clip2(_q10 - Delta1, max_v);

		*p20 = Clip2(_p20 + Delta2, max_v);
		*q20 = Clip2(_q20 - Delta2, max_v);

		*p30 = Clip2(_p30 + Delta3, max_v);
		*q30 = Clip2(_q30 - Delta3, max_v);

		return;
	}
};



int main()
{
	std::cout << "Hello, world" << std::endl;
	std::string input_file_name = "image1_352x288_yv12.yuv";
	std::string output_file_name = "image1_filtered_352x288_yv12.yuv";
	unsigned int width = 352;
	unsigned int height = 288;
	unsigned int Qp = 40;

	ReadYuvFrame frame(input_file_name.c_str(), width, height, Qp);
	frame.DeblockingFilter();
	frame.Save(output_file_name.c_str());

    return 0;
}

