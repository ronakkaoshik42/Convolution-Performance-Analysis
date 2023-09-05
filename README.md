# Convolution-Performance-Analysis (CPU vs. GPU)

This project demonstrates the comparison between CPU and GPU implementations of convolution using C++. It includes a CPU-based convolution in C++ and a GPU-based convolution in CUDA. You can use this project to compare the execution times and results of these two implementations.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- CMake (version 3.15 or higher)
- CUDA Toolkit (for GPU support)
- C++11 compatible compiler
- NVIDIA GPU with CUDA support (for GPU execution)

## Build Instructions

Clone the repository & build the project:

   ```bash
   git clone https://github.com/yourusername/convolution-comparison.git
   cd convolution-comparison
   mkdir build
   cd build
   cmake ..
   make
   ```
## Running the Code

### CPU Convolution

To run the CPU-based convolution, use the following command:
   ```bash
   ./main <rows> <cols>
   ```
Replace rows and cols with the desired dimensions of the input matrix.

### GPU Convolution

To run the GPU-based convolution, use the following command:
   ```bash
   ./main <rows> <cols>
   ```
