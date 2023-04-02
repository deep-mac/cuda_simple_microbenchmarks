/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
//#include <cuda_pipeline_primitives.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#define BLOCK_X 32
#define BLOCK_Y 4
#define TOTAL_THREADS BLOCK_X*BLOCK_Y
#define OUTER_LOOP 1e6
#define INNER_LOOP 5 //make this 40 to get all misses in L1 but hits in L2
#define L2_FACTOR 32
#define STRIDE 1
#define ARR_SIZE BLOCK_X*BLOCK_Y*INNER_LOOP*STRIDE*L2_FACTOR

template <int BLOCK_SIZE> __global__ void readWriteCUDA_L1(int *arr, int *out, uint64_t *clockStart, uint64_t *clockEnd) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    clockStart[ty*BLOCK_X+tx] = clock64();
    if (bx == 0 && by == 0){
        #pragma unroll
        for (long int j = 0; j < OUTER_LOOP; j++){
            //printf("Inside if2 \n");
            for (int i = 0; i < INNER_LOOP; i++){
                //out[(ty*32+tx)*INNER_LOOP+i] = arr[(ty*32+tx)*INNER_LOOP+i]; //This generates some striding issue 
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //This generates all misses in L1 but hits in L2
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //Just L1, L2 involved
                out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; 
            }
            __syncthreads();
            //atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
        }
    }
    clockEnd[ty*BLOCK_X+tx] = clock64();
    //printf("SYNC: %ld\n", *clock);

    //__syncthreads();
}

template <int BLOCK_SIZE> __global__ void readWriteCUDA_L2(int *arr, int *out, uint64_t *clockStart, uint64_t *clockEnd) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    clockStart[ty*BLOCK_X+tx] = clock64();
    if (bx == 0 && by == 0){
        #pragma unroll
        for (long int j = 0; j < OUTER_LOOP; j++){
            //printf("Inside if2 \n");
            for (int i = 0; i < INNER_LOOP*L2_FACTOR; i++){
                //out[(ty*32+tx)*INNER_LOOP+i] = arr[(ty*32+tx)*INNER_LOOP+i]; //This generates some striding issue 
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //This generates all misses in L1 but hits in L2
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //Just L1, L2 involved
                out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; 
            }
            __syncthreads();
            //atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
        }
    }
    clockEnd[ty*BLOCK_X+tx] = clock64();
    //printf("SYNC: %ld\n", *clock);

    //__syncthreads();
}


template <int BLOCK_SIZE> __global__ void readWriteCUDA(int *arr, int *out, uint64_t *clockStart, uint64_t *clockEnd) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ int var[INNER_LOOP][BLOCK_Y][BLOCK_X];

    clockStart[ty*BLOCK_X+tx] = clock64();
    if (bx == 0 && by == 0){
        #pragma unroll
        for (long int j = 0; j < OUTER_LOOP; j++){
            //printf("Inside if2 \n");
            for (int i = 0; i < INNER_LOOP; i++){
                //out[(ty*32+tx)*INNER_LOOP+i] = arr[(ty*32+tx)*INNER_LOOP+i]; //This generates some striding issue 
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //This generates all misses in L1 but hits in L2
                //out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; //Just L1, L2 involved
                var[i][ty][tx] = arr[i*TOTAL_THREADS*STRIDE+(ty*32+tx)]; 
            }
            __syncthreads();
            //atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
            for (int i = 0; i < INNER_LOOP; i++){
                out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = var[i][ty][tx];
            }
        }
    }
    clockEnd[ty*BLOCK_X+tx] = clock64();
    //printf("SYNC: %ld\n", *clock);

    //__syncthreads();
}

__global__ void readWriteCUDA_async(int *arr, int *out, uint64_t *clockStart, uint64_t *clockEnd) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ int var[INNER_LOOP*BLOCK_Y*BLOCK_X];
    auto block = cooperative_groups::this_thread_block();
    //printf("block.size = %d\n", block.size());
    // Create a synchronization object (C++20 barrier)
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0) {
        init(&barrier, block.size()); // Friend function initializes barrier
    }
    block.sync();

    clockStart[ty*BLOCK_X+tx] = clock64();
    if (bx == 0 && by == 0){
        #pragma unroll
        for (long int j = 0; j < OUTER_LOOP; j++){
            /*for (size_t i = 0; i < INNER_LOOP; i++) {
                __pipeline_memcpy_async(&var[i][ty][tx], &arr[i*TOTAL_THREADS+(ty*32+tx)], sizeof(int));
            }*/
            /*__pipeline_memcpy_async(&var, &arr, sizeof(int)*block.size());
            __pipeline_commit();
            __pipeline_wait_prior(0);*/
            cuda::memcpy_async(block, var, arr, sizeof(int) * block.size()*INNER_LOOP, barrier);
            barrier.arrive_and_wait(); // Waits for all copies to complete
            //atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
            block.sync();
            for (int i = 0; i < INNER_LOOP; i++){
                out[i*TOTAL_THREADS*STRIDE+(ty*32+tx)] = var[i*TOTAL_THREADS+(ty*32+tx)];
            }
        }
    }
    clockEnd[ty*BLOCK_X+tx] = clock64();
    //printf("ASYNC: %ld\n", *clock);
    //__syncthreads();
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}
void ConstantInitInt(int *data, int size, int val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

void ConstantInitInt(uint64_t *data, int size, int val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}


void printClock(uint64_t *hClockStart, uint64_t *hClockEnd){
    uint64_t minStart = hClockStart[0];
    uint64_t maxEnd = hClockEnd[0];
    for (int i = 0; i<BLOCK_Y; i++){
        for (int j = 0; j<BLOCK_X; j++){
            if (minStart > hClockStart[i*BLOCK_X+j])
                minStart = hClockStart[i*BLOCK_X+j];
            if (maxEnd < hClockEnd[i*BLOCK_X+j])
                maxEnd = hClockEnd[i*BLOCK_X+j];
            //printf("ty = %d, tx = %d, clock start = %ld, clock end = %ld, diff = %ld\n", i, j, hClockStart[i*BLOCK_X+j], hClockEnd[i*BLOCK_X+j], hClockEnd[i*BLOCK_X+j]-hClockStart[i*BLOCK_X+j]);
        }
    }
    printf("MIN START = %ld, MAX END = %ld, diff = %ld\n", minStart, maxEnd, maxEnd-minStart);
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int ReadWrite(int argc, char **argv, int block_size, int block_size_y, const dim3 &dimsA, const dim3 &dimsB) {

    cudaStream_t stream;
    cudaEvent_t start, stop;

    //Create event and stream
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


    dim3 grid(1, 1);
    dim3 threads(block_size, block_size_y);
    int *arrH, *arrD, *out, *outH;
    uint64_t *clockStart, *clockEnd, *hClockStart, *hClockEnd;

    //Host mallocs
    checkCudaErrors(cudaMallocHost(&arrH, ARR_SIZE*sizeof(int)));
    checkCudaErrors(cudaMallocHost(&outH, ARR_SIZE*sizeof(int)));
    cudaMallocHost(&hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y);
    cudaMallocHost(&hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y);
    
    //Device mallocs
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&arrD), ARR_SIZE *sizeof(int)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&out), ARR_SIZE *sizeof(int)));
    cudaMalloc(&clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y);
    cudaMalloc(&clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y);

    //Host init
    ConstantInitInt(arrH, ARR_SIZE, 2);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);

    //Host to device copy
    checkCudaErrors(cudaMemcpyAsync(arrD, arrH, ARR_SIZE*sizeof(int), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        readWriteCUDA_L1<16>
            <<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
    } else {
        readWriteCUDA_L1<32>
            <<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
    }

    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaMemcpy(hClockStart, clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(hClockEnd, clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    printf("warmup done\n");
    printClock(hClockStart, hClockEnd);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));

    // Record the start event
    float msecTotal = 0.0f;
    int nIter = 1;
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            readWriteCUDA_L1<16>
                <<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        } else {
            readWriteCUDA_L1<32>
                <<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        }
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal/INNER_LOOP/OUTER_LOOP/nIter/block_size/block_size_y;
    cudaMemcpy(hClockStart, clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(hClockEnd, clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    printf("L1 Time= %.9f nsec\n", msecPerMatrixMul*1e6);
    printClock(hClockStart, hClockEnd);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));

    //New kernel
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            readWriteCUDA_L2<16><<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        } else {
            readWriteCUDA_L2<32><<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        }
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    msecPerMatrixMul = msecTotal/INNER_LOOP/OUTER_LOOP/nIter/block_size/block_size_y/L2_FACTOR;
    cudaMemcpy(hClockStart, clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(hClockEnd, clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    printf("L2 Time= %.9f nsec\n", msecPerMatrixMul*1e6);
    printClock(hClockStart, hClockEnd);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));


    //New kernel
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            readWriteCUDA<16><<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        } else {
            readWriteCUDA<32><<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        }
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    msecPerMatrixMul = msecTotal/INNER_LOOP/OUTER_LOOP/nIter/block_size/block_size_y;
    cudaMemcpy(hClockStart, clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(hClockEnd, clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    printf("Shared Time= %.9f nsec\n", msecPerMatrixMul*1e6);
    printClock(hClockStart, hClockEnd);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));


    //NEW KERNEL
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            readWriteCUDA_async<<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        } else {
            readWriteCUDA_async<<<grid, threads, 0, stream>>>(arrD, out, clockStart, clockEnd);
        }
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    msecPerMatrixMul = msecTotal/INNER_LOOP/OUTER_LOOP/nIter/block_size/block_size_y;
    cudaMemcpy(hClockStart, clockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(hClockEnd, clockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyDeviceToHost);
    printf("Async Time= %.9f nsec\n", msecPerMatrixMul*1e6);
    printClock(hClockStart, hClockEnd);
    ConstantInitInt(hClockStart, ARR_SIZE, 0);
    ConstantInitInt(hClockEnd, ARR_SIZE, 0);
    checkCudaErrors(cudaMemcpyAsync(clockStart, hClockStart, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(clockEnd, hClockEnd, sizeof(uint64_t)*BLOCK_X*BLOCK_Y, cudaMemcpyHostToDevice, stream));

    // Clean up memory
    checkCudaErrors(cudaFreeHost(arrH));
    checkCudaErrors(cudaFree(arrD));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return EXIT_SUCCESS;
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
                " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = BLOCK_X;
    int block_size_y = BLOCK_Y;

    checkCudaErrors(cudaProfilerStart());
    int result = ReadWrite(argc, argv, block_size, block_size_y, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(result);
}

