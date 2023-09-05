#include <iostream>
#include <chrono>
#include <random>

// CUDA header
#include <cuda_runtime.h>
#include <cuda.h>

// Function to get the current timestamp
std::chrono::high_resolution_clock::time_point getCurrentTime()
{
    return std::chrono::high_resolution_clock::now();
}

// Function to compute the time difference in milliseconds
double getElapsedTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Function to perform convolution along the rows (horizontal) on GPU
__global__ void HorizontalConvolutionGPU(const unsigned char *input, int rows, int cols, int *output)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows)
    {
        if (j == 0)
        {
            output[i * cols] = input[i * cols + 1];
        }
        else if (j < cols - 1)
        {
            output[i * cols + j] = input[i * cols + j + 1] - input[i * cols + j - 1];
        }
        else
        {
            output[i * cols + cols - 1] = -input[i * cols + cols - 2];
        }
    }
}

// Function to perform convolution along the columns (vertical) on GPU
__global__ void VerticalConvolutionGPU(const unsigned char *input, int rows, int cols, int *output)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < cols)
    {
        if (i == 0)
        {
            output[j] = input[cols + j];
        }
        else if (i < rows - 1)
        {
            output[i * cols + j] = input[(i + 1) * cols + j] - input[(i - 1) * cols + j];
        }
        else
        {
            output[(rows - 1) * cols + j] = -input[(rows - 2) * cols + j];
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <rows> <cols>\n";
        return 1;
    }

    // Define the size of the memory to allocate (for example, 1000 integers)
    size_t size = 1000 * sizeof(int);

    // Pointer to GPU memory
    int *gpuData;

    // Allocate memory on the GPU
    cudaError_t cudaStatus = cudaMalloc((void **)&gpuData, size);

    // Check if memory allocation was successful
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "CUDA memory allocation error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error as needed
    }

    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);

    // Host (CPU) memory allocation and initialization
    unsigned char *matrix = new unsigned char[rows * cols];
    int *Dx = new int[rows * cols];
    int *Dy = new int[rows * cols];

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned char> dist(0, std::numeric_limits<unsigned char>::max());

    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = dist(rng);
    }

    // Device (GPU) memory allocation
    unsigned char *d_matrix;
    int *d_Dx;
    int *d_Dy;

    cudaMalloc((void **)&d_matrix, sizeof(unsigned char) * rows * cols);
    cudaMalloc((void **)&d_Dx, sizeof(int) * rows * cols);
    cudaMalloc((void **)&d_Dy, sizeof(int) * rows * cols);

    // Copy data from host to device
    cudaMemcpy(d_matrix, matrix, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice);

    // Define thread and block dimensions for CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Perform horizontal convolution on GPU
    auto startTime = getCurrentTime();
    HorizontalConvolutionGPU<<<gridSize, blockSize>>>(d_matrix, rows, cols, d_Dx);
    cudaDeviceSynchronize();
    auto endTime = getCurrentTime();
    double horizontalTime = getElapsedTime(startTime, endTime);

    // Perform vertical convolution on GPU
    startTime = getCurrentTime();
    VerticalConvolutionGPU<<<gridSize, blockSize>>>(d_matrix, rows, cols, d_Dy);
    cudaDeviceSynchronize();
    endTime = getCurrentTime();
    double verticalTime = getElapsedTime(startTime, endTime);

    // Copy results from device to host
    cudaMemcpy(Dx, d_Dx, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
    cudaMemcpy(Dy, d_Dy, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

    // Find the min and max values for Dx and Dy
    int minDx = Dx[0];
    int maxDx = Dx[0];
    int minDy = Dy[0];
    int maxDy = Dy[0];

    for (int i = 0; i < rows * cols; ++i)
    {
        minDx = std::min(minDx, Dx[i]);
        maxDx = std::max(maxDx, Dx[i]);
        minDy = std::min(minDy, Dy[i]);
        maxDy = std::max(maxDy, Dy[i]);
    }

    std::cout << "Horizontal Convolution Time (GPU): " << horizontalTime << " ms\n";
    std::cout << "Vertical Convolution Time (GPU): " << verticalTime << " ms\n";
    std::cout << "Min Dx: " << minDx << " Max Dx: " << maxDx << "\n";
    std::cout << "Min Dy: " << minDy << " Max Dy: " << maxDy << "\n";

    // Clean up memory
    delete[] matrix;
    delete[] Dx;
    delete[] Dy;
    cudaFree(d_matrix);
    cudaFree(d_Dx);
    cudaFree(d_Dy);

    return 0;
}
