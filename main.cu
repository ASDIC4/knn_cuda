#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>

// constant definitions 
const int DIMENSIONS = 100;         // 增加维度到100
//const int NUM_POINTS = 1000000;     // 100万个数据点
//const int NUM_QUERIES = 1000;       // 1000个查询点

const int NUM_POINTS = 10000;     // 100万个数据点
const int NUM_QUERIES = 100;       // 1000个查询点

const int K = 50;                   // 找出50个最近邻


const int BLOCK_SIZE = 256;

// CUDA错误检查实用函数
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// CUDA核函数：计算距离
__device__ double enhancedDistanceGPU(const double* a, const double* b) {
    double sum = 0.0;
    double weight;

    for (int i = 0; i < DIMENSIONS; ++i) {
        weight = 1.0 + sin(i * M_PI / DIMENSIONS);
        double diff = a[i] - b[i];
        sum += weight * (diff * diff);

        if (i % 2 == 0) {
            sum += weight * abs(sin(diff));
        }
        else {
            sum += weight * abs(cos(diff));
        }
    }
    return sqrt(sum);
}

// CUDA核函数：计算单个查询点的所有距离
__global__ void computeDistances(const double* data, const double* query,
    double* distances, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        distances[idx] = enhancedDistanceGPU(&data[idx * DIMENSIONS], query);
    }
}

// 数据生成函数
void generateData(const std::string& filename) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::ofstream file(filename);
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            file << dis(gen);
            if (j < DIMENSIONS - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

// 数据加载函数
void loadData(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> point;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value));
        }
        data.push_back(point);
    }
    file.close();
}

// 生成查询点
std::vector<std::vector<double>> generateQueryPoints() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<std::vector<double>> queryPoints(NUM_QUERIES, std::vector<double>(DIMENSIONS));
    for (int i = 0; i < NUM_QUERIES; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            queryPoints[i][j] = dis(gen);
        }
    }
    return queryPoints;
}

// CPU端的主要处理函数
void knnGPU(const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& queries,
    std::vector<std::vector<std::pair<int, double>>>& results) {
    // 分配和准备GPU内存
    double* d_data, * d_query, * d_distances;
    int* d_indices;

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, NUM_POINTS * DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_query, DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, NUM_POINTS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_indices, NUM_POINTS * sizeof(int)));

    // 将数据转换为连续数组并复制到GPU
    std::vector<double> flatData(NUM_POINTS * DIMENSIONS);
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            flatData[i * DIMENSIONS + j] = data[i][j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, flatData.data(),
        NUM_POINTS * DIMENSIONS * sizeof(double),
        cudaMemcpyHostToDevice));

    // 为每个查询点计算KNN
    int numBlocks = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int q = 0; q < queries.size(); ++q) {
        // 复制查询点到GPU
        CHECK_CUDA_ERROR(cudaMemcpy(d_query, queries[q].data(),
            DIMENSIONS * sizeof(double),
            cudaMemcpyHostToDevice));

        // 计算距离
        computeDistances <<<numBlocks, BLOCK_SIZE >>> (d_data, d_query,
            d_distances, NUM_POINTS);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 复制距离回主机
        std::vector<double> distances(NUM_POINTS);
        CHECK_CUDA_ERROR(cudaMemcpy(distances.data(), d_distances,
            NUM_POINTS * sizeof(double),
            cudaMemcpyDeviceToHost));

        // 在CPU上执行部分排序（对于小的k值这样更快）
        std::vector<std::pair<double, int>> distanceIndex(NUM_POINTS);
        for (int i = 0; i < NUM_POINTS; ++i) {
            distanceIndex[i] = { distances[i], i };
        }
        std::partial_sort(distanceIndex.begin(),
            distanceIndex.begin() + K,
            distanceIndex.end());

        // 保存结果
        results[q].resize(K);
        for (int i = 0; i < K; ++i) {
            results[q][i] = { distanceIndex[i].second, distanceIndex[i].first };
        }
    }

    // 释放GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_query));
    CHECK_CUDA_ERROR(cudaFree(d_distances));
    CHECK_CUDA_ERROR(cudaFree(d_indices));
}

int main() {
    // 生成数据
    std::string filename = "knn_large_data.txt";
    generateData(filename);

    // 加载数据
    std::vector<std::vector<double>> data;
    std::cout << "Loading data..." << std::endl;
    loadData(filename, data);

    // 生成查询点
    std::vector<std::vector<double>> queryPoints = generateQueryPoints();

    // 准备结果容器
    std::vector<std::vector<std::pair<int, double>>> results(NUM_QUERIES);

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 执行GPU版本的KNN
    knnGPU(data, queryPoints, results);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出结果
    std::cout << "GPU Processing time: " << duration.count() << " ms" << std::endl;

    // 输出部分结果示例
    std::cout << "\nFirst query point results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Index: " << results[0][i].first
            << " Distance: " << results[0][i].second << std::endl;
    }

    return 0;
}
