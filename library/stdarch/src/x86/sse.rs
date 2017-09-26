use simd_llvm::simd_shuffle4;
use v128::*;

#[cfg(test)]
use assert_instr::assert_instr;

/// Adds the first component of `a` and `b`, the other components are copied
/// from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(addss))]
pub fn _mm_add_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { addss(a, b) }
}

/// Adds f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(addps))]
pub fn _mm_add_ps(a: f32x4, b: f32x4) -> f32x4 {
    a + b
}

/// Subtracts the first component of `b` from `a`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(subss))]
pub fn _mm_sub_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { subss(a, b) }
}

/// Subtracts f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(subps))]
pub fn _mm_sub_ps(a: f32x4, b: f32x4) -> f32x4 {
    a - b
}

/// Multiplies the first component of `a` and `b`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(mulss))]
pub fn _mm_mul_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { mulss(a, b) }
}

/// Multiplies f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(mulps))]
pub fn _mm_mul_ps(a: f32x4, b: f32x4) -> f32x4 {
    a * b
}

/// Divides the first component of `b` by `a`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(divss))]
pub fn _mm_div_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { divss(a, b) }
}

/// Divides f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(divps))]
pub fn _mm_div_ps(a: f32x4, b: f32x4) -> f32x4 {
    a / b
}

/// Return the square root of the first single-precision (32-bit)
/// floating-point element in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(sqrtss))]
pub fn _mm_sqrt_ss(a: f32x4) -> f32x4 {
    unsafe { sqrtss(a) }
}

/// Return the square root of packed single-precision (32-bit) floating-point
/// elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(sqrtps))]
pub fn _mm_sqrt_ps(a: f32x4) -> f32x4 {
    unsafe { sqrtps(a) }
}

/// Return the approximate reciprocal of the first single-precision
/// (32-bit) floating-point element in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rcpss))]
pub fn _mm_rcp_ss(a: f32x4) -> f32x4 {
    unsafe { rcpss(a) }
}

/// Return the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rcpps))]
pub fn _mm_rcp_ps(a: f32x4) -> f32x4 {
    unsafe { rcpps(a) }
}

/// Return the approximate reciprocal square root of the fist single-precision
/// (32-bit) floating-point elements in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rsqrtss))]
pub fn _mm_rsqrt_ss(a: f32x4) -> f32x4 {
    unsafe { rsqrtss(a) }
}

/// Return the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rsqrtps))]
pub fn _mm_rsqrt_ps(a: f32x4) -> f32x4 {
    unsafe { rsqrtps(a) }
}

/// Compare the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the minimum value in the first element of the return 
/// value, the other elements are copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(minss))]
pub fn _mm_min_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { minss(a, b) }
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding minimum values.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(minps))]
pub fn _mm_min_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { minps(a, b) }
}

/// Compare the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the maximum value in the first element of the return 
/// value, the other elements are copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maxss))]
pub fn _mm_max_ss(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { maxss(a, b) }
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding maximum values.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maxps))]
pub fn _mm_max_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { maxps(a, b) }
}

// Shuffle packed single-precision (32-bit) floating-point elements in `a` and `b`
// using `mask`.
// The lower half of result takes values from `a` and the higher half from `b`.
// Mask is split to 2 control bits each to index the element from inputs.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(shufps))]
pub fn _mm_shuffle_ps(a: f32x4, b: f32x4, mask: i32) -> f32x4 {
    let mask = (mask & 0xFF) as u8;
    
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            unsafe {
                simd_shuffle4(a, b, [$x01, $x23, $x45, $x67])
            }
        }
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (mask >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 4),
                0b01 => shuffle_done!($x01, $x23, $x45, 5),
                0b10 => shuffle_done!($x01, $x23, $x45, 6),
                _ => shuffle_done!($x01, $x23, $x45, 7),
            }
        }
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (mask >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 4),
                0b01 => shuffle_x67!($x01, $x23, 5),
                0b10 => shuffle_x67!($x01, $x23, 6),
                _ => shuffle_x67!($x01, $x23, 7),
            }
        }
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (mask >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        }
    }
    match mask & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    }
}

/// Unpack and interleave single-precision (32-bit) floating-point elements
/// from the higher half of `a` and `b`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(unpckhps))]
pub fn _mm_unpackhi_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { simd_shuffle4(a, b, [2, 6, 3, 7]) }
}

/// Unpack and interleave single-precision (32-bit) floating-point elements
/// from the lower half of `a` and `b`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(unpcklps))]
pub fn _mm_unpacklo_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { simd_shuffle4(a, b, [0, 4, 1, 5]) }
}

/// Combine higher half of `a` and `b`. The highwe half of `b` occupies the lower
/// half of result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(movhlps))]
pub fn _mm_movehl_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { simd_shuffle4(a, b, [6, 7, 2, 3]) }
}

/// Combine lower half of `a` and `b`. The lower half of `b` occupies the higher
/// half of result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(unpcklpd))]
pub fn _mm_movelh_ps(a: f32x4, b: f32x4) -> f32x4 {
    unsafe { simd_shuffle4(a, b, [0, 1, 4, 5]) }
}

/// Return a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 4 least significant bits of the return value.
/// All other bits are set to `0`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(movmskps))]
pub fn _mm_movemask_ps(a: f32x4) -> i32 {
    unsafe { movmskps(a) }
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse.add.ss"]
    fn addss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.sub.ss"]
    fn subss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.mul.ss"]
    fn mulss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.div.ss"]
    fn divss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.sqrt.ss"]
    fn sqrtss(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.sqrt.ps"]
    fn sqrtps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rcp.ss"]
    fn rcpss(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rcp.ps"]
    fn rcpps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rsqrt.ss"]
    fn rsqrtss(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.rsqrt.ps"]
    fn rsqrtps(a: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.min.ss"]
    fn minss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.min.ps"]
    fn minps(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.max.ss"]
    fn maxss(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.max.ps"]
    fn maxps(a: f32x4, b: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse.movmsk.ps"]
    fn movmskps(a: f32x4) -> i32;
}

#[cfg(all(test, target_feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use v128::*;
    use x86::sse;

    #[test]
    #[target_feature = "+sse"]
    fn _mm_add_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_add_ps(a, b);
        assert_eq!(r, f32x4::new(-101.0, 25.0, 0.0, -15.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_add_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_add_ss(a, b);
        assert_eq!(r, f32x4::new(-101.0, 5.0, 0.0, -10.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_sub_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_sub_ps(a, b);
        assert_eq!(r, f32x4::new(99.0, -15.0, 0.0, -5.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_sub_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_sub_ss(a, b);
        assert_eq!(r, f32x4::new(99.0, 5.0, 0.0, -10.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_mul_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_mul_ps(a, b);
        assert_eq!(r, f32x4::new(100.0, 100.0, 0.0, 50.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_mul_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_mul_ss(a, b);
        assert_eq!(r, f32x4::new(100.0, 5.0, 0.0, -10.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_div_ps() {
        let a = f32x4::new(-1.0, 5.0, 2.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.2, -5.0);
        let r = sse::_mm_div_ps(a, b);
        assert_eq!(r, f32x4::new(0.01, 0.25, 10.0, 2.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_div_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_div_ss(a, b);
        assert_eq!(r, f32x4::new(0.01, 5.0, 0.0, -10.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_sqrt_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_sqrt_ss(a);
        let e = f32x4::new(2.0, 13.0, 16.0, 100.0);
        assert_eq!(r, e);
    }

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
    fn _mm_rcp_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rcp_ss(a);
        let e = f32x4::new(0.24993896, 13.0, 16.0, 100.0);
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
    fn _mm_rsqrt_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rsqrt_ss(a);
        let e = f32x4::new(0.49987793, 13.0, 16.0, 100.0);
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
    fn _mm_min_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_min_ss(a, b);
        assert_eq!(r, f32x4::new(-100.0, 5.0, 0.0, -10.0));
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
    fn _mm_max_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_max_ss(a, b);
        assert_eq!(r, f32x4::new(-1.0, 5.0, 0.0, -10.0));
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
    fn _mm_shuffle_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let mask = 0b00_01_01_11;
        let r = sse::_mm_shuffle_ps(a, b, mask);
        assert_eq!(r, f32x4::new(4.0, 2.0, 6.0, 5.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_unpackhi_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_unpackhi_ps(a, b);
        assert_eq!(r, f32x4::new(3.0, 7.0, 4.0, 8.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_unpacklo_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_unpacklo_ps(a, b);
        assert_eq!(r, f32x4::new(1.0, 5.0, 2.0, 6.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_movehl_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_movehl_ps(a, b);
        assert_eq!(r, f32x4::new(7.0, 8.0, 3.0, 4.0));
    }

    #[test]
    #[target_feature = "+sse"]
    fn _mm_movelh_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_movelh_ps(a, b);
        assert_eq!(r, f32x4::new(1.0, 2.0, 5.0, 6.0));
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
