#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 32


#define SUB_MATRIX_THREAD 2
#define ZAD5_BLOCK_SIZE 16
#define MATRIX_SIZE 1024


__global__ void
MatrixMultiplyKernel_GlobalMem(float* C, const float* A, const float* B, unsigned int matrixDim)
{
	unsigned int squareBlockDim = blockDim.x;

	// Compute the row index
	unsigned int i = (squareBlockDim * blockIdx.y) + threadIdx.y;
	// Compute the column index
	unsigned int j = (squareBlockDim * blockIdx.x) + threadIdx.x;

	//unsigned int index = (i * matrixDim) + j;
	float sum = 0.0f;

	for (unsigned int x = i; x < matrixDim; x+=squareBlockDim)
	{
		for (unsigned int y = j; y < matrixDim; y+=squareBlockDim)
		{
			for (unsigned int k = 0; k < matrixDim; ++k)
			{
				sum += A[x * matrixDim + k] * B[k * matrixDim + y];
			}
			C[(x * matrixDim) + y] = sum;
			sum = 0;
		}
	}
}

__global__ void
matrixMulSharedMultiBlock(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
	float Csub = 0;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
	}

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void
matrixMulNeighbours(float *C, float *A, float *B, int wA, int wB)
{
	int subSize = blockDim.x * SUB_MATRIX_THREAD;
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x * SUB_MATRIX_THREAD;
	int ty = threadIdx.y * SUB_MATRIX_THREAD;

	int aBegin = wA * by * subSize;
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = subSize;

	int bBegin = subSize * bx;
	int bStep = subSize * wB;

	float Csub[SUB_MATRIX_THREAD * SUB_MATRIX_THREAD];
	for (int i = 0; i < SUB_MATRIX_THREAD; i++)
	{
		for (int j = 0; j < SUB_MATRIX_THREAD; j++)
		{
			Csub[i * SUB_MATRIX_THREAD + j] = 0;
		}
	}

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	__shared__ float As[ZAD5_BLOCK_SIZE * SUB_MATRIX_THREAD][ZAD5_BLOCK_SIZE * SUB_MATRIX_THREAD];
	__shared__ float Bs[ZAD5_BLOCK_SIZE * SUB_MATRIX_THREAD][ZAD5_BLOCK_SIZE * SUB_MATRIX_THREAD];

	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep)
	{
#pragma unroll
		for (int i = 0; i < SUB_MATRIX_THREAD; ++i)
		{
#pragma unroll
			for (int j = 0; j < SUB_MATRIX_THREAD; ++j)
			{
				As[ty + i][tx + j] = A[a + wA * (ty + i) + tx + j];
				Bs[ty + i][tx + j] = B[b + wB * (ty + i) + tx + j];
			}
		}
		__syncthreads();
#pragma unroll
		for (int i = 0; i < SUB_MATRIX_THREAD; ++i)
		{
#pragma unroll
			for (int j = 0; j < SUB_MATRIX_THREAD; ++j)
			{
#pragma unroll
				for (int k = 0; k < subSize; ++k)
				{
					Csub[i * SUB_MATRIX_THREAD + j] += As[ty + i][k] * Bs[k][tx + j];
				}
			}
		}
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	for (int i = 0; i < SUB_MATRIX_THREAD; ++i)
	{
		for (int j = 0; j < SUB_MATRIX_THREAD; ++j)
		{
			int c = wB * subSize * by + subSize * bx;
			C[c + wB * (ty + i) + tx + j] = Csub[i * SUB_MATRIX_THREAD + j];
		}
	}
}


__global__ void matrixMulGlobalMem(float* Cd, float* Ad, float* Bd, int width)
{
	float tmpC = 0;
#pragma unroll
	for (int ty = threadIdx.y; ty < width; ty += BLOCK_SIZE){
#pragma unroll
		for (int tx = threadIdx.x; tx < width; tx += BLOCK_SIZE){
			tmpC = 0;
#pragma unroll
			for (int k = 0; k < width; ++k){
				tmpC += Ad[ty * width + k] * Bd[k * width + tx];
			}
			Cd[ty * width + tx] = tmpC;
		}
	}
}


void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

#pragma region alloc host memory
	if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
#pragma endregion

    // Setup execution parameters
	//dim3 grid(1, 1); //zad1
    
    //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 grid(dimsB.x / threads.x , dimsA.y / threads.y); // zad3

	dim3 threads(ZAD5_BLOCK_SIZE, ZAD5_BLOCK_SIZE);
	dim3 grid(dimsB.x / (threads.x * SUB_MATRIX_THREAD), dimsA.y / (threads.y * SUB_MATRIX_THREAD));

	cudaDeviceSynchronize();

#pragma region initEvents
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
#pragma endregion

    int nIter = 1;
    for (int j = 0; j < nIter; j++)
    {
		//matrixMulGlobalMem<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x);					//zad1
		//matrixMulSharedMultiBlock<<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);				//zad3
		matrixMulNeighbours <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);		//zad5
	}

#pragma region Error handling
    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
#pragma endregion

#pragma region countPerformance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threads.x * threads.y);
#pragma endregion

#pragma region copy result to host
	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
#pragma endregion

#pragma region verifyResult

	printf("Checking computed result for correctness: ");
	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-6; // machine zero

	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
	{
		double abs_err = fabs(h_C[i] - (dimsA.x * valB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
			correct = false;
		}
	}
#pragma endregion

#pragma region clean
	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	if (correct)
	{
		return EXIT_SUCCESS;
	}
	else
	{
		return EXIT_FAILURE;
	}
#pragma endregion
}

int main(int argc, char **argv)
{
#pragma region Init
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }


	std::cout << "Block size: " << BLOCK_SIZE << "\n";
#pragma endregion

	int matrixDim = MATRIX_SIZE; //zad 1 3
	//int matrixDim = SUB_MATRIX_THREAD * ZAD5_BLOCK_SIZE * 10; //zad 5
    dim3 dimsA(matrixDim, matrixDim, 1);
    dim3 dimsB(matrixDim, matrixDim, 1);
	
#pragma region ReadArgs
    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
    {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
    {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
#pragma endregion

    //int matrix_result = matrixMultiply(argc, argv, BLOCK_SIZE, dimsA, dimsB); //zad 1 3
	int matrix_result = matrixMultiply(argc, argv, ZAD5_BLOCK_SIZE, dimsA, dimsB); //zad5
	//system("pause");
}
