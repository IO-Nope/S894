// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector"}>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <immintrin.h>
#include <cstdarg>
#include <fstream>
#include "../utils.cpp"
const int max_print = 256;
bool scflag = 1;
bool veflag =1;
float cmp1[max_print];
float cmp2[max_print];

void Dprint(const char* format, ...) {
    char buffer[1024];

    va_list args;
    va_start(args, format);

    std::vsnprintf(buffer, sizeof(buffer), format, args);

    va_end(args);

    std::ofstream outfile("tempout.txt", std::ios::app);
    if (outfile.is_open()) {
        outfile << buffer<<"\n"; // 写入文件
        outfile.close();   // 关闭文件
    }
}
void Dprint(float cx, int& count) {
    if (count < max_print) {
        std::ofstream outfile("tempout.txt", std::ios::app);
        if (outfile.is_open()) {
            if(!(count%8))outfile << "\n";
            outfile << count<<":" <<cx << " "; 
            outfile.close();     
        }
        count++;
    }
}
void Dprint( __m256 cx_vec, int& count) {
    const int step = 8;
    for (int i = 0; i < step; i++) {
        if (count < max_print) {
            float cx = ((float*)&cx_vec)[i]; 
            std::ofstream outfile("tempout.txt", std::ios::app);
            if (outfile.is_open()) {
                if(!(count%8))outfile << "\n";
                outfile << count <<":"<< cx << " "; 
                outfile.close();      
            }
            count++;
        }
    }
}
// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // int count = 0;
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {

            // Get the plane coordinate X for the image pixel.
            float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;
            float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;
            //if(scflag&&!j)Dprint(cy,count);
            // Innermost loop: start the recursion from z = 0.
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
                // if(scflag)Dprint(x,count);
            }

            // Write result.
            //if(scflag)Dprint(iters,count);
            out[i * img_size + j] = iters;
        }
    }
    // if(scflag)Dprint("\nscalar结束\n");
    // scflag =0;
}

/// <--- your code here --->

void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // TODO: Implement this function.
    // int count = 0;
    for(uint64_t i = 0; i< img_size ; ++i){
        for ( uint64_t j = 0; j < img_size ; j+=8){
            __m256 vecx = _mm256_set1_ps((float(j) / float(img_size)) * 2.5f - 2.0f);
            __m256 vecy = _mm256_set1_ps((float(i) / float(img_size)) * 2.5f - 1.25f);
            __m256 vectemp = _mm256_set1_ps((1.0f/float(img_size))* 2.5f);  
            __m256 fc = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
            vectemp = _mm256_mul_ps(vectemp,fc);
            vecx = _mm256_add_ps(vectemp,vecx);
            //if(veflag&&!j)Dprint((vecy)[0],count);

            __m256 vecx2 = _mm256_set1_ps(0.0f);
            __m256 vecy2 = _mm256_set1_ps(0.0f);
            __m256 w = _mm256_set1_ps(0.0f);
            __m256 iters = _mm256_set1_ps(0.0f);       
            const __m256 vect = _mm256_set1_ps(4.0f);
            const __m256 vecm = _mm256_set1_ps(max_iters);
            const __m256 veco = _mm256_set1_ps(1.0f);
            __m256 sum = _mm256_add_ps(vecx2,vecy2);
            uint8_t k1 =AVX2_COMPARE_MASK(sum,vect,_CMP_LE_OS);
            uint8_t k2 =AVX2_COMPARE_MASK(iters,vecm,_CMP_LT_OS);
            uint8_t k = k1&k2;
            while (k>0b0)
            {
                __m256 x= _mm256_set1_ps(0.0f);
                __m256 y = _mm256_set1_ps(0.0f);


                x = masked_sub_ps(x,k,vecx2,vecy2);
                x = masked_add_ps(x,k,x,vecx);
                y = masked_sub_ps(y,k,w,vecx2);
                y = masked_sub_ps(y,k,y,vecy2);
                y = masked_add_ps(y,k,y,vecy);

                vecx2 = masked_mul_ps(vecx2,k,x,x);
                vecy2 = masked_mul_ps(vecy2,k,y,y);
                __m256 z = _mm256_add_ps(x,y);
                w = masked_mul_ps(w,k,z,z);
                iters = masked_add_ps(iters,k,iters,veco);
                sum = _mm256_add_ps(vecx2,vecy2);
                k1 =AVX2_COMPARE_MASK(sum,vect,_CMP_LE_OS);
                k2 =AVX2_COMPARE_MASK(iters,vecm,_CMP_LT_OS);
                k = k1&k2;
                // if(veflag)Dprint(x,count);
            }
            float temp[8];
            //if(veflag)Dprint(iters,count);
            _mm256_storeu_ps(temp,iters);
            for(uint8_t t = 0;t<8;++t){
                out[i * img_size + j +t]= temp[t];
            }
        }
    }
    // if(veflag)Dprint("\nvector 结束\n");
    // veflag =0;
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl { SCALAR, VECTOR, ALL };

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
          } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "测试图片大小为 " << *img_size << "x" << *img_size << " 和 "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter));
                auto intensity = static_cast<uint8_t>(
                    log_iter * 222 / log2f(static_cast<float>(max_iters)));
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << "正在运行 " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::cout << "  运行时间: " << times[0] / 1'000'000 << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 256;
    uint32_t max_iters = 1000;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);
        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
