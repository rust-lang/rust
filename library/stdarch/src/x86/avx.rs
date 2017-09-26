use v256::*;

// #[cfg(test)]
// use assert_instr::assert_instr;

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
pub fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    a + b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
pub fn _mm256_add_ps(a: f32x8, b: f32x8) -> f32x8 {
    a + b
}

/// Alternatively add and subtract packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
pub fn _mm256_addsub_pd(a: f64x4, b: f64x4) -> f64x4 {
    unsafe { addsubpd256(a, b) }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.addsub.pd.256"]
    fn addsubpd256(a: f64x4, b: f64x4) -> f64x4;
}

/// Subtract packed double-precision (64-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
// #[cfg_attr(test, assert_instr(subpd))]
pub fn _mm256_sub_pd(a: f64x4, b: f64x4) -> f64x4 {
    unsafe { subpd256(a, b) }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.sub.pd.256"]
    fn subpd256(a: f64x4, b: f64x4) -> f64x4;
}

/// Subtract packed single-precision (32-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
// #[cfg_attr(test, assert_instr(subps))]
pub fn _mm256_sub_ps(a: f32x8, b: f32x8) -> f32x8 {
    unsafe { subps256(a, b) }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.sub.ps.256"]
    fn subps256(a: f32x8, b: f32x8) -> f32x8;
}

/// Round packed double-precision (64-bit) floating point elements in `a`
/// according to the flag `b`. The value of `b` may be as follows:
///    Bits [7:4] are reserved.
///    Bit [3] is a precision exception value:
///      0: A normal PE exception is used.
///      1: The PE field is not updated.
///    Bit [2] is the rounding control source:
///      0: Use bits [1:0] of \a M.
///      1: Use the current MXCSR setting.
///    Bits [1:0] contain the rounding control definition:
///      00: Nearest.
///      01: Downward (toward negative infinity).
///      10: Upward (toward positive infinity).
///      11: Truncated.
#[inline(always)]
#[target_feature = "+avx"]
// #[cfg_attr(test, assert_instr(vroundpd))]
pub fn _mm256_round_pd(a: f64x4, b: i32) -> f64x4 {
    unsafe { roundpd256(a, b) }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.round.pd.256"]
    fn roundpd256(a: f64x4, b: i32) -> f64x4;
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
// #[cfg_attr(test, assert_instr(vroundpd))]
pub fn _mm256_ceil_pd(a: f64x4) -> f64x4 {
    _mm256_round_pd(a, 0b00000010)
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
// #[cfg_attr(test, assert_instr(vroundpd))]
pub fn _mm256_floor_pd(a: f64x4) -> f64x4 {
    _mm256_round_pd(a, 0b00000001)
}


#[cfg(all(test, target_feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use v256::*;
    use x86::avx;

    #[test]
    #[target_feature = "+avx"]
    fn _mm256_add_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_add_pd(a, b);
        let e = f64x4::new(6.0, 8.0, 10.0, 12.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx"]
    fn _mm256_add_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = avx::_mm256_add_ps(a, b);
        let e = f32x8::new(10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx"]
    fn _mm256_addsub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_addsub_pd(a, b);
        let e = f64x4::new(-4.0,8.0,-4.0,12.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx"]
    fn _mm256_sub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_sub_pd(a, b);
        let e = f64x4::new(-4.0,-4.0,-4.0,-4.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx"]
    fn _mm256_sub_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0);
        let r = avx::_mm256_sub_ps(a, b);
        let e = f32x8::new(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx"]
    pub fn _mm256_round_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_closest = avx::_mm256_round_pd(a, 0b00000000);
        let result_down = avx::_mm256_round_pd(a, 0b00000001);
        let result_up = avx::_mm256_round_pd(a, 0b00000010);
        let expected_closest = f64x4::new(2.0, 2.0, 4.0, -1.0);
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[test]
    #[target_feature = "+avx"]
    pub fn _mm256_floor_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_down = avx::_mm256_floor_pd(a);
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        assert_eq!(result_down, expected_down);
    }

    #[test]
    #[target_feature = "+avx"]
    pub fn _mm256_ceil_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_up = avx::_mm256_ceil_pd(a, );
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_up, expected_up);
    }

}
