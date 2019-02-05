use core_arch::x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512IFMA52&expand=3488)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm512_madd52hi_epu64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    vpmadd52huq_512(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=3497&avx512techs=AVX512IFMA52)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm512_madd52lo_epu64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    vpmadd52luq_512(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=vpmadd52&avx512techs=AVX512IFMA52,AVX512VL&expand=3485)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm256_madd52hi_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52huq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=vpmadd52&avx512techs=AVX512IFMA52,AVX512VL&expand=3494)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm256_madd52lo_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52luq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=3488,3482&text=vpmadd52&avx512techs=AVX512IFMA52,AVX512VL)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm_madd52hi_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52huq_128(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=3488,3491&text=vpmadd52&avx512techs=AVX512IFMA52,AVX512VL)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm_madd52lo_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52luq_128(a, b, c)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.128"]
    fn vpmadd52luq_128(z: __m128i, x: __m128i, y: __m128i) -> __m128i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.128"]
    fn vpmadd52huq_128(z: __m128i, x: __m128i, y: __m128i) -> __m128i;
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.256"]
    fn vpmadd52luq_256(z: __m256i, x: __m256i, y: __m256i) -> __m256i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.256"]
    fn vpmadd52huq_256(z: __m256i, x: __m256i, y: __m256i) -> __m256i;
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.512"]
    fn vpmadd52luq_512(z: __m512i, x: __m512i, y: __m512i) -> __m512i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.512"]
    fn vpmadd52huq_512(z: __m512i, x: __m512i, y: __m512i) -> __m512i;
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use core_arch::x86::*;

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_madd52hi_epu64() {
        let mut a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        a = _mm512_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm512_set1_epi64(11030549757952);

        assert_eq_m512i(a, expected);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_madd52lo_epu64() {
        let mut a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        a = _mm512_madd52lo_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm512_set1_epi64(100055558127628);

        assert_eq_m512i(a, expected);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_madd52hi_epu64() {
        let mut a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        a = _mm256_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm256_set1_epi64x(11030549757952);

        assert_eq_m256i(a, expected);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_madd52lo_epu64() {
        let mut a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        a = _mm256_madd52lo_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm256_set1_epi64x(100055558127628);

        assert_eq_m256i(a, expected);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_madd52hi_epu64() {
        let mut a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        a = _mm_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm_set1_epi64x(11030549757952);

        assert_eq_m128i(a, expected);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_madd52lo_epu64() {
        let mut a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        a = _mm_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm_set1_epi64x(11030549757952);

        assert_eq_m128i(a, expected);
    }
}
