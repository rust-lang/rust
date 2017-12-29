//! `x86_64`'s Streaming SIMD Extensions 2 (SSE2)

use v128::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse2.cvtsd2si64"]
    fn cvtsd2si64(a: f64x2) -> i64;
    #[link_name = "llvm.x86.sse2.cvttsd2si64"]
    fn cvttsd2si64(a: f64x2) -> i64;
}

/// Convert the lower double-precision (64-bit) floating-point element in a to
/// a 64-bit integer.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvtsd2si))]
pub unsafe fn _mm_cvtsd_si64(a: f64x2) -> i64 {
    cvtsd2si64(a)
}

/// Alias for [`_mm_cvtsd_si64`](fn._mm_cvtsd_si64_ss.html).
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvtsd2si))]
pub unsafe fn _mm_cvtsd_si64x(a: f64x2) -> i64 {
    _mm_cvtsd_si64(a)
}

/// Convert the lower double-precision (64-bit) floating-point element in `a`
/// to a 64-bit integer with truncation.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvttsd2si))]
pub unsafe fn _mm_cvttsd_si64(a: f64x2) -> i64 {
    cvttsd2si64(a)
}

/// Alias for [`_mm_cvttsd_si64`](fn._mm_cvttsd_si64_ss.html).
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvttsd2si))]
pub unsafe fn _mm_cvttsd_si64x(a: f64x2) -> i64 {
    _mm_cvttsd_si64(a)
}

/// Stores a 64-bit integer value in the specified memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be
/// used again soon).
#[inline(always)]
#[target_feature = "+sse2"]
// FIXME movnti on windows and linux x86_64
//#[cfg_attr(test, assert_instr(movntiq))]
pub unsafe fn _mm_stream_si64(mem_addr: *mut i64, a: i64) {
    ::core::intrinsics::nontemporal_store(mem_addr, a);
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(windows)), assert_instr(movq))]
pub unsafe fn _mm_cvtsi64_si128(a: i64) -> i64x2 {
    i64x2::new(a, 0)
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(windows)), assert_instr(movq))]
pub unsafe fn _mm_cvtsi64x_si128(a: i64) -> i64x2 {
    _mm_cvtsi64_si128(a)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(windows)), assert_instr(movq))]
pub unsafe fn _mm_cvtsi128_si64(a: i64x2) -> i64 {
    a.extract(0)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(windows)), assert_instr(movq))]
pub unsafe fn _mm_cvtsi128_si64x(a: i64x2) -> i64 {
    _mm_cvtsi128_si64(a)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::x86_64::sse2;

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsd_si64() {
        use std::{f64, i64};

        let r = sse2::_mm_cvtsd_si64(f64x2::new(-2.0, 5.0));
        assert_eq!(r, -2_i64);

        let r = sse2::_mm_cvtsd_si64(f64x2::new(f64::MAX, f64::MIN));
        assert_eq!(r, i64::MIN);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsd_si64x() {
        use std::{f64, i64};

        let r = sse2::_mm_cvtsd_si64x(f64x2::new(f64::NAN, f64::NAN));
        assert_eq!(r, i64::MIN);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvttsd_si64() {
        let a = f64x2::new(-1.1, 2.2);
        let r = sse2::_mm_cvttsd_si64(a);
        assert_eq!(r, -1_i64);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvttsd_si64x() {
        use std::{f64, i64};

        let a = f64x2::new(f64::NEG_INFINITY, f64::NAN);
        let r = sse2::_mm_cvttsd_si64x(a);
        assert_eq!(r, i64::MIN);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_stream_si64() {
        let a: i64 = 7;
        let mut mem = ::std::boxed::Box::<i64>::new(-1);
        sse2::_mm_stream_si64(&mut *mem as *mut i64, a);
        assert_eq!(a, *mem);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_si128() {
        let r = sse2::_mm_cvtsi64_si128(5);
        assert_eq!(r, i64x2::new(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi128_si64() {
        let r = sse2::_mm_cvtsi128_si64(i64x2::new(5, 0));
        assert_eq!(r, 5);
    }
}
