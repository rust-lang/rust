#[cfg(test)]
use stdsimd_test::assert_instr;

use v256::*;

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    a + b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm256_add_ps(a: f32x8, b: f32x8) -> f32x8 {
    a + b
}

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmulpd))]
pub unsafe fn _mm256_mul_pd(a: f64x4, b: f64x4) -> f64x4 {
    a * b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmulps))]
pub unsafe fn _mm256_mul_ps(a: f32x8, b: f32x8) -> f32x8 {
    a * b
}

/// Alternatively add and subtract packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddsubpd))]
pub unsafe fn _mm256_addsub_pd(a: f64x4, b: f64x4) -> f64x4 {
    addsubpd256(a, b)
}

/// Alternatively add and subtract packed single-precision (32-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddsubps))]
pub unsafe fn _mm256_addsub_ps(a: f32x8, b: f32x8) -> f32x8 {
    addsubps256(a, b)
}

/// Subtract packed double-precision (64-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsubpd))]
pub unsafe fn _mm256_sub_pd(a: f64x4, b: f64x4) -> f64x4 {
    a - b
}

/// Subtract packed single-precision (32-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsubps))]
pub unsafe fn _mm256_sub_ps(a: f32x8, b: f32x8) -> f32x8 {
    a - b
}

/// Round packed double-precision (64-bit) floating point elements in `a`
/// according to the flag `b`. The value of `b` may be as follows:
///
/// ```ignore
/// 0x00: Round to the nearest whole number.
/// 0x01: Round down, toward negative infinity.
/// 0x02: Round up, toward positive infinity.
/// 0x03: Truncate the values.
/// ```
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_round_pd(a: f64x4, b: i32) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => { roundpd256(a, $imm8) }
    }
    constify_imm8!(b, call)
}

#[cfg(test)]
#[cfg_attr(test, assert_instr(vroundpd))]
#[target_feature = "+avx"]
fn test_mm256_round_pd(a: f64x4) -> f64x4 {
    unsafe { _mm256_round_pd(a, 0x3) }
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd))]
pub unsafe fn _mm256_ceil_pd(a: f64x4) -> f64x4 {
    roundpd256(a, 0x02)
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// negative infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd))]
pub unsafe fn _mm256_floor_pd(a: f64x4) -> f64x4 {
    roundpd256(a, 0x01)
}

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.addsub.pd.256"]
    fn addsubpd256(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.addsub.ps.256"]
    fn addsubps256(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.round.pd.256"]
    fn roundpd256(a: f64x4, b: i32) -> f64x4;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v256::*;
    use x86::avx;

    #[simd_test = "avx"]
    fn _mm256_add_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = unsafe { avx::_mm256_add_pd(a, b) };
        let e = f64x4::new(6.0, 8.0, 10.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_add_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = unsafe { avx::_mm256_add_ps(a, b) };
        let e = f32x8::new(10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_mul_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = unsafe { avx::_mm256_mul_pd(a, b) };
        let e = f64x4::new(5.0, 12.0, 21.0, 32.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_mul_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = unsafe { avx::_mm256_mul_ps(a, b) };
        let e = f32x8::new(9.0, 20.0, 33.0, 48.0, 65.0, 84.0, 105.0, 128.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_addsub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = unsafe { avx::_mm256_addsub_pd(a, b) };
        let e = f64x4::new(-4.0, 8.0, -4.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_addsub_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0);
        let r = unsafe { avx::_mm256_addsub_ps(a, b) };
        let e = f32x8::new(-4.0, 8.0, -4.0, 12.0, -4.0, 8.0, -4.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_sub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = unsafe { avx::_mm256_sub_pd(a, b) };
        let e = f64x4::new(-4.0,-4.0,-4.0,-4.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_sub_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0);
        let r = unsafe { avx::_mm256_sub_ps(a, b) };
        let e = f32x8::new(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_round_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_closest = unsafe { avx::_mm256_round_pd(a, 0b00000000) };
        let result_down = unsafe { avx::_mm256_round_pd(a, 0b00000001) };
        let result_up = unsafe { avx::_mm256_round_pd(a, 0b00000010) };
        let expected_closest = f64x4::new(2.0, 2.0, 4.0, -1.0);
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    fn _mm256_floor_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_down = unsafe { avx::_mm256_floor_pd(a) };
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        assert_eq!(result_down, expected_down);
    }

    #[simd_test = "avx"]
    fn _mm256_ceil_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_up = unsafe { avx::_mm256_ceil_pd(a) };
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_up, expected_up);
    }
}
