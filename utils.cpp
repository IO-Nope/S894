#include <immintrin.h>
#include <stdint.h>

// 将 __mmask8 转换为 AVX2 掩码向量
inline __m256 mask_to_m256(__mmask8 mask) {
    // 将掩码的每一位扩展为 32 位整数
    __m256i mask_epi32 = _mm256_set_epi32(
        (mask & 0x80) ? -1 : 0,
        (mask & 0x40) ? -1 : 0,
        (mask & 0x20) ? -1 : 0,
        (mask & 0x10) ? -1 : 0,
        (mask & 0x08) ? -1 : 0,
        (mask & 0x04) ? -1 : 0,
        (mask & 0x02) ? -1 : 0,
        (mask & 0x01) ? -1 : 0
    );
    // 转换为浮点向量
    return _mm256_castsi256_ps(mask_epi32);
}


inline __m256 masked_add_ps(__m256 src, __mmask8 mask, __m256 a, __m256 b) {
    __m256 mask_vec = mask_to_m256(mask); 
    __m256 result = _mm256_add_ps(a, b);
    return _mm256_blendv_ps(src, result, mask_vec); 
}

inline __m256 masked_sub_ps(__m256 src, __mmask8 mask, __m256 a, __m256 b) {
    __m256 mask_vec = mask_to_m256(mask);
    __m256 result = _mm256_sub_ps(a, b); 
    return _mm256_blendv_ps(src, result, mask_vec);
}


inline __m256 masked_mul_ps(__m256 src, __mmask8 mask, __m256 a, __m256 b) {
    __m256 mask_vec = mask_to_m256(mask); 
    __m256 result = _mm256_mul_ps(a, b); 
    return _mm256_blendv_ps(src, result, mask_vec); 
}


inline __m256 masked_div_ps(__m256 src, __mmask8 mask, __m256 a, __m256 b) {
    __m256 mask_vec = mask_to_m256(mask); 
    __m256 result = _mm256_div_ps(a, b); 
    return _mm256_blendv_ps(src, result, mask_vec); 
}

// 定义宏，用于比较并返回 8 位整数掩码
#define AVX2_COMPARE_MASK(a, b, cmp_op) (static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps((a), (b), (cmp_op))))) 
void print_m256(__m256 vec) {
    float values[8];
    _mm256_storeu_ps(values, vec);
    for (int i = 0; i < 8; i++) {
        printf("%f ", values[i]);
    }
}