cmake_minimum_required(VERSION 3.15)
project(ConvolutionComparison CUDA CXX)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Add the main.cpp file
add_executable(main main.cpp)

# Add the convolution.cu file (GPU code)
cuda_add_executable(convolution convolution.cu)

# Specify CUDA architecture options for your GPU (e.g., sm_30 for Kepler)
set_target_properties(convolution PROPERTIES CUDA_ARCHITECTURES "sm_30")

# Link CUDA runtime library for the convolution target
target_link_libraries(convolution CUDA::cudart)

# Link the CPU and GPU targets against appropriate libraries
target_link_libraries(main)
target_link_libraries(convolution)

