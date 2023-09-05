#include <iostream>
#include <chrono>
#include <random>

// Compiling and running
// g++ -O3 -std=c++11 -o convolution convolution.cpp
// ./convolution 100 100

// Assumptions :
// - The input matrix has sufficient memory allocation (>50000*50000 fails) available for both horizontal and vertical convolutions.
// - The input matrix is stored in row-major order, with each row having 'cols' elements (for 1D contiguous storage to improve data access).

// Optimizations implemented :
// - Nested Loop Ordering: Optimizes data access by looping over rows before columns (horizontal) and vice versa (vertical).
// - Cache Locality: Enhances memory access performance by using row-major order for 1D representation of the 2D matrix.
// - Minimization of Boundary Checks: Unnecessary boundary checks removed by spliting the for loops for edges.
// - Loop unrolling implemented through the compiler optimization flags.

// Function to get the current timestamp
std::chrono::high_resolution_clock::time_point getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

// Function to compute the time difference in milliseconds
double getElapsedTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Generic function to print a 2D matrix (represented as 1D contiguous memory)
template <typename T>
void printMatrixGeneric(const T* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << static_cast<int>(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


// Function to perform convolution along the rows (horizontal)
void HorizontalConvolution(const unsigned char* input, int rows, int cols, int* output) {
    for (int i = 0; i < rows; ++i) {
        // Convolution for the first column (j = 0)
        output[i * cols] = input[i * cols + 1];

        // Convolution for the remaining columns (1 to cols-2)
        for (int j = 1; j < cols - 1; ++j) {
            output[i * cols + j] = input[i * cols + j + 1] - input[i * cols + j - 1];
        }

        // Convolution for the last column (j = cols-1)
        output[i * cols + cols - 1] = -input[i * cols + cols - 2];
    }
}

// Function to perform convolution along the columns (vertical)
void VerticalConvolution(const unsigned char* input, int rows, int cols, int* output) {
    // Convolution for the first row (i = 0)
    for (int j = 0; j < cols; ++j) {
        output[j] = input[cols + j];
    }

    // Convolution for the remaining rows (1 to rows-2)
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i * cols + j] = input[(i + 1) * cols + j] - input[(i - 1) * cols + j];
        }
    }

    // Convolution for the last row (i = rows-1)
    for (int j = 0; j < cols; ++j) {
        output[(rows - 1) * cols + j] = -input[(rows - 2) * cols + j];
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <rows> <cols>\n";
        return 1;
    }

    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);

    alignas(64) unsigned char* matrix = new unsigned char[rows * cols];
    alignas(64) int* Dx = new int[rows * cols];
    alignas(64) int* Dy = new int[rows * cols];

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned char> dist(0, std::numeric_limits<unsigned char>::max());

    // Initialize the input matrix with random unsigned char values.
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(rng);
    }

    auto startTime = getCurrentTime();
    HorizontalConvolution(matrix, rows, cols, Dx);
    auto endTime = getCurrentTime();
    double horizontalTime = getElapsedTime(startTime, endTime);

    startTime = getCurrentTime();
    VerticalConvolution(matrix, rows, cols, Dy);
    endTime = getCurrentTime();
    double verticalTime = getElapsedTime(startTime, endTime);

    // Find the min and max values for Dx and Dy
    int minDx = Dx[0];
    int maxDx = Dx[0];
    int minDy = Dy[0];
    int maxDy = Dy[0];

    for (int i = 0; i < rows * cols; ++i) {
        minDx = std::min(minDx, Dx[i]);
        maxDx = std::max(maxDx, Dx[i]);
        minDy = std::min(minDy, Dy[i]);
        maxDy = std::max(maxDy, Dy[i]);
    }

    std::cout << "Horizontal Convolution Time: " << horizontalTime << " ms\n";
    std::cout << "Vertical Convolution Time: " << verticalTime << " ms\n";
    std::cout << "Min Dx: " << minDx << " Max Dx: " << maxDx << "\n";
    std::cout << "Min Dy: " << minDy << " Max Dy: " << maxDy << "\n";


    // Uncomment to print the matrices
    // std::cout << "Input Matrix:\n";
    // printMatrixGeneric(matrix, rows, cols);

    // // Call the function to print the Dx and Dy matrices
    // std::cout << "Dx Matrix:\n";
    // printMatrixGeneric(Dx, rows, cols);

    // std::cout << "Dy Matrix:\n";
    // printMatrixGeneric(Dy, rows, cols);

    // Clean up memory
    delete[] matrix;
    delete[] Dx;
    delete[] Dy;

    return 0;
}

// #include <immintrin.h> // Include the header for AVX2/SSE instructions as well as use the march=native compiler flag
// Small improvement observed (~5-7%) with SIMD enabled which could be attributed to scheduling uncertainty as well so discraded this approach

// void HorizontalConvolution_SIMD(const unsigned char* input, int rows, int cols, int* output) {
//     for (int i = 0; i < rows; ++i) {
//         // Convolution for the first column (j = 0)
//         output[i * cols] = input[i * cols + 1];

//         // Convolution for the remaining columns (1 to cols-2)
//         for (int j = 1; j < cols - 1; j += 256) {
//             __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i * cols + j - 1]));
//             __m256i next_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i * cols + j + 1]));

//             __m256i result = _mm256_sub_epi8(next_data, data);

//             _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[i * cols + j]), result);
//         }

//         // Convolution for the last column (j = cols-1)
//         output[i * cols + cols - 1] = -input[i * cols + cols - 2];
//     }
// }
