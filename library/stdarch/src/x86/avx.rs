use v256::*;

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
    fn addsubpd256(a: f64x4, b:f64x4) -> f64x4;
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
        let r = avx::_mm256_add_pd(a, b);
        let e = f64x4::new(6.0, 8.0, 10.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_add_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = avx::_mm256_add_ps(a, b);
        let e = f32x8::new(10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    fn _mm256_addsub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_addsub_pd(a, b);
        let e = f64x4::new(-4.0,8.0,-4.0,12.0);
        assert_eq!(r, e);
    }
}
