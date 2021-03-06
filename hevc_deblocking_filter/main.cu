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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "hevc_deblocking_filter_cpu.h"

using namespace std;

void ExecuteCpu(std::string const &input_file_name, std::string const &output_file_name,
	unsigned int width, unsigned int height, unsigned int Qp) {

	// one-thread CPU mode run
	ReadYuvFrame frame(input_file_name.c_str(), width, height, Qp);
	auto start = std::chrono::system_clock::now();
	frame.DeblockingFilter();
	auto end = std::chrono::system_clock::now();
	frame.Save(output_file_name.c_str());
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Elapsed Time for one-thread CPU implementation: " << elapsed_seconds.count() << "s" << std::endl;

	// two-thread CPU mode run
	ReadYuvFrame frame2(input_file_name.c_str(), width, height, Qp);
	start = std::chrono::system_clock::now();
	frame2.DeblockingFilter(2);
	end = std::chrono::system_clock::now();
	frame2.Save(output_file_name.c_str());
	elapsed_seconds = end - start;
	std::cout << "Elapsed Time for two-thread CPU implementation: " << elapsed_seconds.count() << "s" << std::endl;

	// four-thread CPU mode run
	ReadYuvFrame frame4(input_file_name.c_str(), width, height, Qp);
	start = std::chrono::system_clock::now();
	frame4.DeblockingFilter(4);
	end = std::chrono::system_clock::now();
	frame4.Save(output_file_name.c_str());
	elapsed_seconds = end - start;
	std::cout << "Elapsed Time for four-thread CPU implementation: " << elapsed_seconds.count() << "s" << std::endl;

	// six-thread CPU mode run
	ReadYuvFrame frame6(input_file_name.c_str(), width, height, Qp);
	start = std::chrono::system_clock::now();
	frame6.DeblockingFilter(6);
	end = std::chrono::system_clock::now();
	frame6.Save(output_file_name.c_str());
	elapsed_seconds = end - start;
	std::cout << "Elapsed Time for six-thread CPU implementation: " << elapsed_seconds.count() << "s" << std::endl;

	// eight-thread CPU mode run
	ReadYuvFrame frame8(input_file_name.c_str(), width, height, Qp);
	start = std::chrono::system_clock::now();
	frame8.DeblockingFilter(8);
	end = std::chrono::system_clock::now();
	frame8.Save(output_file_name.c_str());
	elapsed_seconds = end - start;
	std::cout << "Elapsed Time for eight-thread CPU implementation: " << elapsed_seconds.count() << "s" << std::endl;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void ExecuteGpu(std::string const &input_file_name, std::string const &output_file_name,
	unsigned int width, unsigned int height, unsigned int Qp,
	unsigned dimx1 = 16, unsigned int dimy1 = 16, unsigned dimx2 = 16, unsigned int dimy2 = 16
	);

void GetGpuDeviceInfo() {
	int iDev = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, iDev);

	printf("==============================================\n");
	printf("Device %d: %s\n", iDev, iProp.name);
	printf("Compute Capability: %d.%d\n", iProp.major, iProp.minor);
	printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
	printf("Total amount of constant memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
	printf("Total amount of global memory: %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
	printf("Total amount of shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
	printf("Warp size: %d\n", iProp.warpSize);
	printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
	printf("==============================================\n\n\n");
}

int main()
{
	// input parameters
	//std::string input_file_name = "image1_352x288_yv12.yuv";
	//std::string output_file_name = "image1_filtered_352x288_yv12.yuv";
	//std::string output_file_name_gpu = "image1_filtered_352x288_yv12_gpu.yuv";
	//unsigned int width = 352;
	//unsigned int height = 288;
	//unsigned int Qp = 30;

	//image2_768x576.yuv
	//std::string input_file_name = "image2_768x576.yuv";
	//std::string output_file_name = "image2_filtered_768x576.yuv";
	//std::string output_file_name_gpu = "image2_filtered_768x576_gpu.yuv";
	//unsigned int width = 768;
	//unsigned int height = 576;
	//unsigned int Qp = 30;

	// mother-daughter_352x288_yv12.yuv
	std::string input_file_name = "mother-daughter_352x288_yv12.yuv";
	std::string output_file_name = "mother-daughter_352x288_yv12_filtered.yuv";
	std::string output_file_name_gpu = "mother-daughter_352x288_yv12_filtered_gpu.yuv";
	unsigned int width = 352;
	unsigned int height = 288;
	unsigned int Qp = 35;

	GetGpuDeviceInfo();

	ExecuteCpu(input_file_name, output_file_name, width, height, Qp);
	ExecuteGpu(input_file_name, output_file_name_gpu, width, height, Qp, 20, 20, 20, 20);

	return 0;
}
