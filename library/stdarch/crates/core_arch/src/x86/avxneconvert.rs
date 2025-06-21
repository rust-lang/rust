use crate::arch::asm;
use crate::core_arch::x86::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Convert scalar BF16 (16-bit) floating point element stored at memory locations starting at location
/// a to single precision (32-bit) floating-point, broadcast it to packed single precision (32-bit)
/// floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_bcstnebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vbcstnebf162ps))]
#[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
pub unsafe fn _mm_bcstnebf16_ps(a: *const bf16) -> __m128 {
    bcstnebf162ps_128(a)
}

/// Convert scalar BF16 (16-bit) floating point element stored at memory locations starting at location
/// a to single precision (32-bit) floating-point, broadcast it to packed single precision (32-bit) floating-point
/// elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_bcstnebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vbcstnebf162ps))]
#[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
pub unsafe fn _mm256_bcstnebf16_ps(a: *const bf16) -> __m256 {
    bcstnebf162ps_256(a)
}

/// Convert scalar half-precision (16-bit) floating-point element stored at memory locations starting
/// at location a to a single-precision (32-bit) floating-point, broadcast it to packed single-precision
/// (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_bcstnesh_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vbcstnesh2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_bcstnesh_ps(a: *const f16) -> __m128 {
    bcstnesh2ps_128(a)
}

/// Convert scalar half-precision (16-bit) floating-point element stored at memory locations starting
/// at location a to a single-precision (32-bit) floating-point, broadcast it to packed single-precision
/// (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_bcstnesh_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vbcstnesh2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_bcstnesh_ps(a: *const f16) -> __m256 {
    bcstnesh2ps_256(a)
}

/// Convert packed BF16 (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneebf162ps))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_cvtneebf16_ps(a: *const __m128bh) -> __m128 {
    transmute(cvtneebf162ps_128(a))
}

/// Convert packed BF16 (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneebf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneebf162ps))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_cvtneebf16_ps(a: *const __m256bh) -> __m256 {
    transmute(cvtneebf162ps_256(a))
}

/// Convert packed half-precision (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneeph_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneeph2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_cvtneeph_ps(a: *const __m128h) -> __m128 {
    transmute(cvtneeph2ps_128(a))
}

/// Convert packed half-precision (16-bit) floating-point even-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneeph_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneeph2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_cvtneeph_ps(a: *const __m256h) -> __m256 {
    transmute(cvtneeph2ps_256(a))
}

/// Convert packed BF16 (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneobf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneobf162ps))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_cvtneobf16_ps(a: *const __m128bh) -> __m128 {
    transmute(cvtneobf162ps_128(a))
}

/// Convert packed BF16 (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneobf16_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneobf162ps))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_cvtneobf16_ps(a: *const __m256bh) -> __m256 {
    transmute(cvtneobf162ps_256(a))
}

/// Convert packed half-precision (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneoph_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneoph2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_cvtneoph_ps(a: *const __m128h) -> __m128 {
    transmute(cvtneoph2ps_128(a))
}

/// Convert packed half-precision (16-bit) floating-point odd-indexed elements stored at memory locations starting at
/// location a to single precision (32-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneoph_ps)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneoph2ps))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_cvtneoph_ps(a: *const __m256h) -> __m256 {
    transmute(cvtneoph2ps_256(a))
}

/// Convert packed single precision (32-bit) floating-point elements in a to packed BF16 (16-bit) floating-point
/// elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneps_avx_pbh)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneps2bf16))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_cvtneps_avx_pbh(a: __m128) -> __m128bh {
    unsafe {
        let mut dst: __m128bh;
        asm!(
            "{{vex}}vcvtneps2bf16 {dst},{src}",
            dst = lateout(xmm_reg) dst,
            src = in(xmm_reg) a,
            options(pure, nomem, nostack, preserves_flags)
        );
        dst
    }
}

/// Convert packed single precision (32-bit) floating-point elements in a to packed BF16 (16-bit) floating-point
/// elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtneps_avx_pbh)
#[inline]
#[target_feature(enable = "avxneconvert")]
#[cfg_attr(test, assert_instr(vcvtneps2bf16))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm256_cvtneps_avx_pbh(a: __m256) -> __m128bh {
    unsafe {
        let mut dst: __m128bh;
        asm!(
            "{{vex}}vcvtneps2bf16 {dst},{src}",
            dst = lateout(xmm_reg) dst,
            src = in(ymm_reg) a,
            options(pure, nomem, nostack, preserves_flags)
        );
        dst
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.vbcstnebf162ps128"]
    fn bcstnebf162ps_128(a: *const bf16) -> __m128;
    #[link_name = "llvm.x86.vbcstnebf162ps256"]
    fn bcstnebf162ps_256(a: *const bf16) -> __m256;
    #[link_name = "llvm.x86.vbcstnesh2ps128"]
    fn bcstnesh2ps_128(a: *const f16) -> __m128;
    #[link_name = "llvm.x86.vbcstnesh2ps256"]
    fn bcstnesh2ps_256(a: *const f16) -> __m256;

    #[link_name = "llvm.x86.vcvtneebf162ps128"]
    fn cvtneebf162ps_128(a: *const __m128bh) -> __m128;
    #[link_name = "llvm.x86.vcvtneebf162ps256"]
    fn cvtneebf162ps_256(a: *const __m256bh) -> __m256;
    #[link_name = "llvm.x86.vcvtneeph2ps128"]
    fn cvtneeph2ps_128(a: *const __m128h) -> __m128;
    #[link_name = "llvm.x86.vcvtneeph2ps256"]
    fn cvtneeph2ps_256(a: *const __m256h) -> __m256;

    #[link_name = "llvm.x86.vcvtneobf162ps128"]
    fn cvtneobf162ps_128(a: *const __m128bh) -> __m128;
    #[link_name = "llvm.x86.vcvtneobf162ps256"]
    fn cvtneobf162ps_256(a: *const __m256bh) -> __m256;
    #[link_name = "llvm.x86.vcvtneoph2ps128"]
    fn cvtneoph2ps_128(a: *const __m128h) -> __m128;
    #[link_name = "llvm.x86.vcvtneoph2ps256"]
    fn cvtneoph2ps_256(a: *const __m256h) -> __m256;
}

#[cfg(test)]
mod tests {
    use crate::core_arch::simd::{u16x4, u16x8};
    use crate::core_arch::x86::*;
    use crate::mem::transmute_copy;
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
        let a = bf16::from_bits(BF16_ONE);
        let r = _mm_bcstnebf16_ps(addr_of!(a));
        let e = _mm_set_ps(1., 1., 1., 1.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_bcstnebf16_ps() {
        let a = bf16::from_bits(BF16_ONE);
        let r = _mm256_bcstnebf16_ps(addr_of!(a));
        let e = _mm256_set_ps(1., 1., 1., 1., 1., 1., 1., 1.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_bcstnesh_ps() {
        let a = 1.0_f16;
        let r = _mm_bcstnesh_ps(addr_of!(a));
        let e = _mm_set_ps(1., 1., 1., 1.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_bcstnesh_ps() {
        let a = 1.0_f16;
        let r = _mm256_bcstnesh_ps(addr_of!(a));
        let e = _mm256_set_ps(1., 1., 1., 1., 1., 1., 1., 1.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneebf16_ps() {
        let a = __m128bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm_cvtneebf16_ps(addr_of!(a));
        let e = _mm_setr_ps(1., 3., 5., 7.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneebf16_ps() {
        let a = __m256bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm256_cvtneebf16_ps(addr_of!(a));
        let e = _mm256_setr_ps(1., 3., 5., 7., 1., 3., 5., 7.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneeph_ps() {
        let a = __m128h([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let r = _mm_cvtneeph_ps(addr_of!(a));
        let e = _mm_setr_ps(1., 3., 5., 7.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneeph_ps() {
        let a = __m256h([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let r = _mm256_cvtneeph_ps(addr_of!(a));
        let e = _mm256_setr_ps(1., 3., 5., 7., 9., 11., 13., 15.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneobf16_ps() {
        let a = __m128bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm_cvtneobf16_ps(addr_of!(a));
        let e = _mm_setr_ps(2., 4., 6., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneobf16_ps() {
        let a = __m256bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm256_cvtneobf16_ps(addr_of!(a));
        let e = _mm256_setr_ps(2., 4., 6., 8., 2., 4., 6., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneoph_ps() {
        let a = __m128h([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let r = _mm_cvtneoph_ps(addr_of!(a));
        let e = _mm_setr_ps(2., 4., 6., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneoph_ps() {
        let a = __m256h([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let r = _mm256_cvtneoph_ps(addr_of!(a));
        let e = _mm256_setr_ps(2., 4., 6., 8., 10., 12., 14., 16.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm_cvtneps_avx_pbh() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let r: u16x4 = transmute_copy(&_mm_cvtneps_avx_pbh(a));
        let e = u16x4::new(BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avxneconvert")]
    unsafe fn test_mm256_cvtneps_avx_pbh() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r: u16x8 = transmute(_mm256_cvtneps_avx_pbh(a));
        let e = u16x8::new(
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        );
        assert_eq!(r, e);
    }
}
