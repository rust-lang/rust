use v128::*;

/// Return the square root of packed single-precision (32-bit) floating-point
/// elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_sqrt_ps(a: f32x4) -> f32x4 {
    unsafe { sqrtps(a) }
}

/// Return the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_rcp_ps(a: f32x4) -> f32x4 {
    unsafe { rcpps(a) }
}

/// Return the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_rsqrt_ps(a: f32x4) -> f32x4 {
    unsafe { rsqrtps(a) }
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding minimum values.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_min_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { minps(a, b) }
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding maximum values.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_max_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { maxps(a, b) }
}

/// Return a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 4 least significant bits of the return value.
/// All other bits are set to `0`.
#[inline(always)]
#[target_feature = "+sse"]
pub fn _mm_movemask_ps(a: f32x4) -> i32 {
    unsafe { movmskps(a) }
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse.sqrt.ps"]
    fn sqrtps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rcp.ps"]
    fn rcpps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rsqrt.ps"]
    fn rsqrtps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.min.ps"]
    fn minps(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.max.ps"]
    fn maxps(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.movmsk.ps"]
    fn movmskps(a: f32x4) -> i32;
}

#[cfg(test)]
mod tests {
    use v128::*;
    use x86::sse;

    #[test]
    #[target_feature = "+sse"]
    fn _mm_sqrt_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_sqrt_ps(a);
        let e = f32x4::new(2.0, 3.6055512, 4.0, 10.0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_rcp_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rcp_ps(a);
        let e = f32x4::new(0.24993896, 0.0769043, 0.06248474, 0.0099983215);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_rsqrt_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rsqrt_ps(a);
        let e = f32x4::new(0.49987793, 0.2772827, 0.24993896, 0.099990845);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_min_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_min_ps(a, b);
        assert_eq!(r, f32x4::new(-100.0, 5.0, 0.0, -10.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_max_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_max_ps(a, b);
        assert_eq!(r, f32x4::new(-1.0, 20.0, 0.0, -5.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_movemask_ps() {
        let r = sse::_mm_movemask_ps(f32x4::new(-1.0, 5.0, -5.0, 0.0));
        assert_eq!(r, 0b0101);

        let r = sse::_mm_movemask_ps(f32x4::new(-1.0, -5.0, -5.0, 0.0));
        assert_eq!(r, 0b0111);
    }
}
