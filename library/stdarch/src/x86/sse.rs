use simd_llvm::simd_shuffle4;
use v128::*;
use std::os::raw::c_void;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Adds the first component of `a` and `b`, the other components are copied
/// from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(addss))]
pub unsafe fn _mm_add_ss(a: f32x4, b: f32x4) -> f32x4 {
    addss(a, b)
}

/// Adds f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(addps))]
pub unsafe fn _mm_add_ps(a: f32x4, b: f32x4) -> f32x4 {
    a + b
}

/// Subtracts the first component of `b` from `a`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(subss))]
pub unsafe fn _mm_sub_ss(a: f32x4, b: f32x4) -> f32x4 {
    subss(a, b)
}

/// Subtracts f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(subps))]
pub unsafe fn _mm_sub_ps(a: f32x4, b: f32x4) -> f32x4 {
    a - b
}

/// Multiplies the first component of `a` and `b`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(mulss))]
pub unsafe fn _mm_mul_ss(a: f32x4, b: f32x4) -> f32x4 {
    mulss(a, b)
}

/// Multiplies f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(mulps))]
pub unsafe fn _mm_mul_ps(a: f32x4, b: f32x4) -> f32x4 {
    a * b
}

/// Divides the first component of `b` by `a`, the other components are
/// copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(divss))]
pub unsafe fn _mm_div_ss(a: f32x4, b: f32x4) -> f32x4 {
    divss(a, b)
}

/// Divides f32x4 vectors.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(divps))]
pub unsafe fn _mm_div_ps(a: f32x4, b: f32x4) -> f32x4 {
    a / b
}

/// Return the square root of the first single-precision (32-bit)
/// floating-point element in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(sqrtss))]
pub unsafe fn _mm_sqrt_ss(a: f32x4) -> f32x4 {
    sqrtss(a)
}

/// Return the square root of packed single-precision (32-bit) floating-point
/// elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(sqrtps))]
pub unsafe fn _mm_sqrt_ps(a: f32x4) -> f32x4 {
    sqrtps(a)
}

/// Return the approximate reciprocal of the first single-precision
/// (32-bit) floating-point element in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rcpss))]
pub unsafe fn _mm_rcp_ss(a: f32x4) -> f32x4 {
    rcpss(a)
}

/// Return the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rcpps))]
pub unsafe fn _mm_rcp_ps(a: f32x4) -> f32x4 {
    rcpps(a)
}

/// Return the approximate reciprocal square root of the fist single-precision
/// (32-bit) floating-point elements in `a`, the other elements are unchanged.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rsqrtss))]
pub unsafe fn _mm_rsqrt_ss(a: f32x4) -> f32x4 {
    rsqrtss(a)
}

/// Return the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(rsqrtps))]
pub unsafe fn _mm_rsqrt_ps(a: f32x4) -> f32x4 {
    rsqrtps(a)
}

/// Compare the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the minimum value in the first element of the return
/// value, the other elements are copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(minss))]
pub unsafe fn _mm_min_ss(a: f32x4, b: f32x4) -> f32x4 {
    minss(a, b)
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding minimum values.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(minps))]
pub unsafe fn _mm_min_ps(a: f32x4, b: f32x4) -> f32x4 {
    minps(a, b)
}

/// Compare the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the maximum value in the first element of the return
/// value, the other elements are copied from `a`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maxss))]
pub unsafe fn _mm_max_ss(a: f32x4, b: f32x4) -> f32x4 {
    maxss(a, b)
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding maximum values.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maxps))]
pub unsafe fn _mm_max_ps(a: f32x4, b: f32x4) -> f32x4 {
    maxps(a, b)
}

/// Shuffle packed single-precision (32-bit) floating-point elements in `a` and
/// `b` using `mask`.
///
/// The lower half of result takes values from `a` and the higher half from
/// `b`. Mask is split to 2 control bits each to index the element from inputs.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(shufps, mask = 3))]
pub unsafe fn _mm_shuffle_ps(a: f32x4, b: f32x4, mask: i32) -> f32x4 {
    let mask = (mask & 0xFF) as u8;

    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            simd_shuffle4(a, b, [$x01, $x23, $x45, $x67])
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
pub unsafe fn _mm_unpackhi_ps(a: f32x4, b: f32x4) -> f32x4 {
    simd_shuffle4(a, b, [2, 6, 3, 7])
}

/// Unpack and interleave single-precision (32-bit) floating-point elements
/// from the lower half of `a` and `b`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(unpcklps))]
pub unsafe fn _mm_unpacklo_ps(a: f32x4, b: f32x4) -> f32x4 {
    simd_shuffle4(a, b, [0, 4, 1, 5])
}

/// Combine higher half of `a` and `b`. The highwe half of `b` occupies the lower
/// half of result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(all(test, not(windows)), assert_instr(movhlps))]
#[cfg_attr(all(test, windows), assert_instr(unpckhpd))]
pub unsafe fn _mm_movehl_ps(a: f32x4, b: f32x4) -> f32x4 {
    // TODO; figure why this is a different instruction on Windows?
    simd_shuffle4(a, b, [6, 7, 2, 3])
}

/// Combine lower half of `a` and `b`. The lower half of `b` occupies the higher
/// half of result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(unpcklpd))]
pub unsafe fn _mm_movelh_ps(a: f32x4, b: f32x4) -> f32x4 {
    simd_shuffle4(a, b, [0, 1, 4, 5])
}

/// Return a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 4 least significant bits of the return value.
/// All other bits are set to `0`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(movmskps))]
pub unsafe fn _mm_movemask_ps(a: f32x4) -> i32 {
    movmskps(a)
}


/// See [`_mm_prefetch`](fn._mm_prefetch.html).
pub const _MM_HINT_T0: i8 = 3;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
pub const _MM_HINT_T1: i8 = 2;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
pub const _MM_HINT_T2: i8 = 1;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
pub const _MM_HINT_NTA: i8 = 0;


/// Fetch the cache line that contains address `p` using the given `strategy`.
///
/// The `strategy` must be one of:
///
/// * [`_MM_HINT_T0`](constant._MM_HINT_T0.html): Fetch into all levels of the
///   cache hierachy.
///
/// * [`_MM_HINT_T1`](constant._MM_HINT_T1.html): Fetch into L2 and higher.
///
/// * [`_MM_HINT_T2`](constant._MM_HINT_T2.html): Fetch into L3 and higher or an
///   implementation-specific choice (e.g., L2 if there is no L3).
///
/// * [`_MM_HINT_NTA`](constant._MM_HINT_NTA.html): Fetch data using the
///   non-temporal access (NTA) hint. It may be a place closer than main memory
///   but outside of the cache hierarchy. This is used to reduce access latency
///   without polluting the cache.
///
/// The actual implementation depends on the particular CPU. This instruction
/// is considered a hint, so the CPU is also free to simply ignore the request.
///
/// The amount of prefetched data depends on the cache line size of the specific
/// CPU, but it will be at least 32 bytes.
///
/// Common caveats:
///
/// * Most modern CPUs already automatically prefetch data based on predicted
///   access patterns.
///
/// * Data is usually not fetched if this would cause a TLB miss or a page
///   fault.
///
/// * Too much prefetching can cause unnecessary cache evictions.
///
/// * Prefetching may also fail if there are not enough memory-subsystem
///   resources (e.g., request buffers).
///
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(prefetcht0, strategy = _MM_HINT_T0))]
// #[cfg_attr(test, assert_instr(prefetcht1, strategy = _MM_HINT_T1))]
// #[cfg_attr(test, assert_instr(prefetcht2, strategy = _MM_HINT_T2))]
// #[cfg_attr(test, assert_instr(prefetchnta, strategy = _MM_HINT_NTA))]
pub unsafe fn _mm_prefetch(p: *const c_void, strategy: i8) {
    // The `strategy` must be a compile-time constant, so we use a short form of
    // `constify_imm8!` for now.
    // We use the `llvm.prefetch` instrinsic with `rw` = 0 (read), and
    // `cache type` = 1 (data cache). `locality` is based on our `strategy`.
    macro_rules! pref {
        ($imm8:expr) => {
            match $imm8 {
                0 => prefetch(p, 0, 0, 1),
                1 => prefetch(p, 0, 1, 1),
                2 => prefetch(p, 0, 2, 1),
                _ => prefetch(p, 0, 3, 1),
            }
        }
    }
    pref!(strategy)
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
    #[link_name = "llvm.prefetch"]
    fn prefetch(p: *const c_void, rw: i32, loc: i32, ty: i32);
}

#[cfg(test)]
mod tests {
    use v128::*;
    use x86::sse;
    use stdsimd_test::simd_test;

    #[simd_test = "sse"]
    unsafe fn _mm_add_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_add_ps(a, b);
        assert_eq!(r, f32x4::new(-101.0, 25.0, 0.0, -15.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_add_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_add_ss(a, b);
        assert_eq!(r, f32x4::new(-101.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_sub_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_sub_ps(a, b);
        assert_eq!(r, f32x4::new(99.0, -15.0, 0.0, -5.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_sub_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_sub_ss(a, b);
        assert_eq!(r, f32x4::new(99.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_mul_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_mul_ps(a, b);
        assert_eq!(r, f32x4::new(100.0, 100.0, 0.0, 50.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_mul_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_mul_ss(a, b);
        assert_eq!(r, f32x4::new(100.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_div_ps() {
        let a = f32x4::new(-1.0, 5.0, 2.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.2, -5.0);
        let r = sse::_mm_div_ps(a, b);
        assert_eq!(r, f32x4::new(0.01, 0.25, 10.0, 2.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_div_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_div_ss(a, b);
        assert_eq!(r, f32x4::new(0.01, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_sqrt_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_sqrt_ss(a);
        let e = f32x4::new(2.0, 13.0, 16.0, 100.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_sqrt_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_sqrt_ps(a);
        let e = f32x4::new(2.0, 3.6055512, 4.0, 10.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_rcp_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rcp_ss(a);
        let e = f32x4::new(0.24993896, 13.0, 16.0, 100.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_rcp_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rcp_ps(a);
        let e = f32x4::new(0.24993896, 0.0769043, 0.06248474, 0.0099983215);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_rsqrt_ss() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rsqrt_ss(a);
        let e = f32x4::new(0.49987793, 13.0, 16.0, 100.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_rsqrt_ps() {
        let a = f32x4::new(4.0, 13.0, 16.0, 100.0);
        let r = sse::_mm_rsqrt_ps(a);
        let e = f32x4::new(0.49987793, 0.2772827, 0.24993896, 0.099990845);
        assert_eq!(r, e);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_min_ss(a, b);
        assert_eq!(r, f32x4::new(-100.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_min_ps(a, b);
        assert_eq!(r, f32x4::new(-100.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_max_ss() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_max_ss(a, b);
        assert_eq!(r, f32x4::new(-1.0, 5.0, 0.0, -10.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_max_ps() {
        let a = f32x4::new(-1.0, 5.0, 0.0, -10.0);
        let b = f32x4::new(-100.0, 20.0, 0.0, -5.0);
        let r = sse::_mm_max_ps(a, b);
        assert_eq!(r, f32x4::new(-1.0, 20.0, 0.0, -5.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_shuffle_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let mask = 0b00_01_01_11;
        let r = sse::_mm_shuffle_ps(a, b, mask);
        assert_eq!(r, f32x4::new(4.0, 2.0, 6.0, 5.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_unpackhi_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_unpackhi_ps(a, b);
        assert_eq!(r, f32x4::new(3.0, 7.0, 4.0, 8.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_unpacklo_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_unpacklo_ps(a, b);
        assert_eq!(r, f32x4::new(1.0, 5.0, 2.0, 6.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_movehl_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_movehl_ps(a, b);
        assert_eq!(r, f32x4::new(7.0, 8.0, 3.0, 4.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_movelh_ps() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let r = sse::_mm_movelh_ps(a, b);
        assert_eq!(r, f32x4::new(1.0, 2.0, 5.0, 6.0));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_movemask_ps() {
        let r = sse::_mm_movemask_ps(f32x4::new(-1.0, 5.0, -5.0, 0.0));
        assert_eq!(r, 0b0101);

        let r = sse::_mm_movemask_ps(f32x4::new(-1.0, -5.0, -5.0, 0.0));
        assert_eq!(r, 0b0111);
    }
}
