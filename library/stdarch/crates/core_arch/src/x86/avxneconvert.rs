use crate::core_arch::{simd::*, x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Convert scalar BF16 (16-bit) floating point element stored at memory locations starting at location
/// a to single precision (32-bit) floating-point, broadcast it to packed single precision (32-bit)
/// floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_bcstnebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vbcstnebf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_bcstnebf16_ps(a: *const u16) -> __m128 {
    transmute(bcstnebf162ps_128(a))
}

/// Convert scalar BF16 (16-bit) floating point element stored at memory locations starting at location
/// a to single precision (32-bit) floating-point, broadcast it to packed single precision (32-bit) floating-point
/// elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_bcstnebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vbcstnebf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_bcstnebf16_ps(a: *const u16) -> __m256 {
    transmute(bcstnebf162ps_256(a))
}

/// Convert packed BF16 (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vcvtneebf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtneebf16_ps(a: *const __m128bh) -> __m128 {
    transmute(cvtneebf162ps_128(a))
}

/// Convert packed BF16 (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vcvtneebf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtneebf16_ps(a: *const __m256bh) -> __m256 {
    transmute(cvtneebf162ps_256(a))
}

/// Convert packed BF16 (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneobf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vcvtneobf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtneobf16_ps(a: *const __m128bh) -> __m128 {
    transmute(cvtneobf162ps_128(a))
}

/// Convert packed BF16 (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneobf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vcvtneobf162ps)
)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtneobf16_ps(a: *const __m256bh) -> __m256 {
    transmute(cvtneobf162ps_256(a))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.vbcstnebf162ps128"]
    fn bcstnebf162ps_128(a: *const u16) -> f32x4;
    #[link_name = "llvm.x86.vbcstnebf162ps256"]
    fn bcstnebf162ps_256(a: *const u16) -> f32x8;

    #[link_name = "llvm.x86.vcvtneebf162ps128"]
    fn cvtneebf162ps_128(a: *const __m128bh) -> __m128;
    #[link_name = "llvm.x86.vcvtneebf162ps256"]
    fn cvtneebf162ps_256(a: *const __m256bh) -> __m256;

    #[link_name = "llvm.x86.vcvtneobf162ps128"]
    fn cvtneobf162ps_128(a: *const __m128bh) -> __m128;
    #[link_name = "llvm.x86.vcvtneobf162ps256"]
    fn cvtneobf162ps_256(a: *const __m256bh) -> __m256;
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use std::ptr::addr_of;
    use stdarch_test::simd_test;

    const BF16_ONE: u16 = 0b0_01111111_0000000;
    const BF16_TWO: u16 = 0b0_10000000_0000000;
    const BF16_THREE: u16 = 0b0_10000000_1000000;
    const BF16_FOUR: u16 = 0b0_10000001_0000000;
    const BF16_FIVE: u16 = 0b0_10000001_0100000;
    const BF16_SIX: u16 = 0b0_10000001_1000000;
    const BF16_SEVEN: u16 = 0b0_10000001_1100000;
    const BF16_EIGHT: u16 = 0b0_10000010_0000000;

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_bcstnebf16_ps() {
        let a = BF16_ONE;
        let r = _mm_bcstnebf16_ps(addr_of!(a));
        let e = _mm_set_ps(1., 1., 1., 1.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_bcstnebf16_ps() {
        let a = BF16_ONE;
        let r = _mm256_bcstnebf16_ps(addr_of!(a));
        let e = _mm256_set_ps(1., 1., 1., 1., 1., 1., 1., 1.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneebf16_ps() {
        let a = __m128bh(
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        );
        let r = _mm_cvtneebf16_ps(addr_of!(a));
        let e = _mm_setr_ps(1., 3., 5., 7.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneebf16_ps() {
        let a = __m256bh(
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        );
        let r = _mm256_cvtneebf16_ps(addr_of!(a));
        let e = _mm256_setr_ps(1., 3., 5., 7., 1., 3., 5., 7.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneobf16_ps() {
        let a = __m128bh(
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        );
        let r = _mm_cvtneobf16_ps(addr_of!(a));
        let e = _mm_setr_ps(2., 4., 6., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneobf16_ps() {
        let a = __m256bh(
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        );
        let r = _mm256_cvtneobf16_ps(addr_of!(a));
        let e = _mm256_setr_ps(2., 4., 6., 8., 2., 4., 6., 8.);
        assert_eq_m256(r, e);
    }
}
