#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include "utils.cpp"
#include <bitset>

int main() {
    // 测试 mask_to_m256
    std::cout << "Testing mask_to_m256:" << std::endl;
    __mmask8 mask = 0b10101010; // 测试掩码
    __m256 mask_vec = mask_to_m256(mask);
    print_m256(mask_vec);

    // 测试 masked_add_ps
    std::cout << "Testing masked_add_ps:" << std::endl;
    __m256 src = _mm256_set1_ps(124.0f);
    __m256 a = _mm256_set1_ps(1.0f);
    __m256 b = _mm256_set1_ps(2.0f);
    __m256 result_add = masked_add_ps(src, mask, a, b);
    print_m256(result_add);

    // 测试 masked_sub_ps
    std::cout << "Testing masked_sub_ps:" << std::endl;
    __m256 result_sub = masked_sub_ps(src, mask, src, b);
    print_m256(result_sub);

    // 测试 masked_mul_ps
    std::cout << "Testing masked_mul_ps:" << std::endl;
    __m256 result_mul = masked_mul_ps(src, mask, a, b);
    print_m256(result_mul);

    // 测试 masked_div_ps
    std::cout << "Testing masked_div_ps:" << std::endl;
    __m256 result_div = masked_div_ps(src, mask, a, b);
    print_m256(result_div);

    // 测试 AVX2_COMPARE_MASK
    std::cout << "Testing AVX2_COMPARE_MASK:" << std::endl;
    __m256 c = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    __m256 d = _mm256_set_ps(5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f);
    uint8_t mask_cmp = AVX2_COMPARE_MASK(c, d, _CMP_GT_OS);
    std::cout << "Comparison mask: " << std::bitset<8>(mask_cmp) << std::endl;

    return 0;
}