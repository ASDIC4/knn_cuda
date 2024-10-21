# CUDA K-Nearest Neighbors (KNN) Implementation

## Project Overview
This project is a GPU-accelerated implementation of the K-Nearest Neighbors (KNN) algorithm using CUDA. The purpose of this project is to leverage the parallel processing capabilities of GPUs to efficiently compute nearest neighbors for large datasets. The implementation focuses on using CUDA to optimize computational performance and reduce the overall runtime, especially for high-dimensional data.

## Features
- **GPU-accelerated distance calculation** using CUDA to speed up the KNN process.
- **Enhanced distance metric** that includes periodic weights and non-linear transformations to add flexibility to distance calculation.
- Efficient **memory management** between host (CPU) and device (GPU).
- Customizable **number of neighbors (K)** and **dimensionality of data**.

## File Structure
- **main.cu**: Core CUDA code that contains the GPU implementation of the KNN algorithm.
- **knn_large_data.txt**: Dataset used for testing and running the KNN queries.
- **knn_cuda.sln / knn_cuda.vcxproj**: Visual Studio solution and project files used for building the project.

## Requirements
- **CUDA Toolkit**: CUDA 12.6.
- **NVIDIA GPU**: Nvidia Driver 565.90 (>= required in the Nvidia website).
- **Visual Studio 2022**: v143.

## Setup and Installation
1. **Clone the Repository**
   ```sh
   git clone git@github.com:ASDIC4/knn_cuda.git
   cd knn_cuda
   ```

2. **Build the Project**
   Use Visual Studio to open the `.sln` file or use `nvcc` directly to compile:
   ```sh
   nvcc main.cu -o knn_cuda
   ```

3. **Run the Application**
   After building, run the executable to test the KNN algorithm:
   ```sh
   ./knn_cuda
   ```

## Implementation Overview
The CUDA implementation of KNN focuses on parallelizing the distance computation for each query point. The key steps include:

1. **Data Generation**: Random data points are generated and saved to a file, which is then used as the reference dataset for KNN queries. The dimensionality and number of points are configurable.

2. **GPU Memory Management**: The dataset and query points are transferred to GPU memory using `cudaMalloc` and `cudaMemcpy`. Efficient memory management is essential for minimizing data transfer time and ensuring smooth operation.

3. **Parallel Distance Calculation**: The CUDA kernel (`computeDistances`) calculates the distance between each query point and all data points concurrently. Each thread is responsible for computing the distance between a single data point and a query point, leveraging the parallel nature of GPUs.

4. **KNN Search**: Once distances are calculated, they are transferred back to host memory where the CPU performs a partial sort to determine the `K` nearest neighbors.

5. **Result Output**: The indices and distances of the nearest neighbors for each query point are outputted, along with the total runtime of the GPU computation.

## CUDA Optimization Techniques
- **Thread Block Configuration**: The code uses a configurable block size (`BLOCK_SIZE`) to maximize GPU core utilization. Choosing an appropriate block size helps achieve efficient parallel execution.
- **Memory Coalescing**: Data is structured to ensure that global memory accesses are coalesced, reducing latency and improving memory bandwidth utilization.
- **Kernel Synchronization**: `cudaDeviceSynchronize()` is used to ensure all GPU computations are completed before transferring data back to the CPU.

## Performance Analysis
The CUDA implementation provides significant speedup compared to a CPU-only implementation, particularly for large datasets and high-dimensional data. By parallelizing the distance calculation, the overall runtime is greatly reduced, making the KNN algorithm feasible for real-time applications in big data environments.
