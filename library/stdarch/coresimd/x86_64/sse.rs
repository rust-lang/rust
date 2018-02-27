//! `x86_64` Streaming SIMD Extensions (SSE)

use coresimd::x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse.cvtss2si64"]
    fn cvtss2si64(a: __m128) -> i64;
    #[link_name = "llvm.x86.sse.cvttss2si64"]
    fn cvttss2si64(a: __m128) -> i64;
    #[link_name = "llvm.x86.sse.cvtsi642ss"]
    fn cvtsi642ss(a: __m128, b: i64) -> __m128;
}

/// Convert the lowest 32 bit float in the input vector to a 64 bit integer.
///
/// The result is rounded according to the current rounding mode. If the result
/// cannot be represented as a 64 bit integer the result will be
/// `0x8000_0000_0000_0000` (`std::i64::MIN`) or trigger an invalid operation
/// floating point exception if unmasked (see
/// [`_mm_setcsr`](fn._mm_setcsr.html)).
///
/// This corresponds to the `CVTSS2SI` instruction (with 64 bit output).
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtss2si))]
pub unsafe fn _mm_cvtss_si64(a: __m128) -> i64 {
    cvtss2si64(a)
}

/// Convert the lowest 32 bit float in the input vector to a 64 bit integer
/// with truncation.
///
/// The result is rounded always using truncation (round towards zero). If the
/// result cannot be represented as a 64 bit integer the result will be
/// `0x8000_0000_0000_0000` (`std::i64::MIN`) or an invalid operation floating
/// point exception if unmasked (see [`_mm_setcsr`](fn._mm_setcsr.html)).
///
/// This corresponds to the `CVTTSS2SI` instruction (with 64 bit output).
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvttss2si))]
pub unsafe fn _mm_cvttss_si64(a: __m128) -> i64 {
    cvttss2si64(a)
}

/// Convert a 64 bit integer to a 32 bit float. The result vector is the input
/// vector `a` with the lowest 32 bit float replaced by the converted integer.
///
/// This intrinsic corresponds to the `CVTSI2SS` instruction (with 64 bit
/// input).
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtsi2ss))]
pub unsafe fn _mm_cvtsi64_ss(a: __m128, b: i64) -> __m128 {
    cvtsi642ss(a, b)
}

#[cfg(test)]
mod tests {
    use std::f32::NAN;
    use std::i64::MIN;

    use stdsimd_test::simd_test;

    use coresimd::arch::x86_64::*;

    #[simd_test = "sse"]
    unsafe fn test_mm_cvtss_si64() {
        let inputs = &[
            (42.0f32, 42i64),
            (-31.4, -31),
            (-33.5, -34),
            (-34.5, -34),
            (4.0e10, 40_000_000_000),
            (4.0e-10, 0),
            (NAN, MIN),
            (2147483500.1, 2147483520),
            (9.223371e18, 9223370937343148032),
        ];
        for i in 0..inputs.len() {
            let (xi, e) = inputs[i];
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvtss_si64(x);
            assert_eq!(
                e, r,
                "TestCase #{} _mm_cvtss_si64({:?}) = {}, expected: {}",
                i, x, r, e
            );
        }
    }

    #[simd_test = "sse"]
    unsafe fn test_mm_cvttss_si64() {
        let inputs = &[
            (42.0f32, 42i64),
            (-31.4, -31),
            (-33.5, -33),
            (-34.5, -34),
            (10.999, 10),
            (-5.99, -5),
            (4.0e10, 40_000_000_000),
            (4.0e-10, 0),
            (NAN, MIN),
            (2147483500.1, 2147483520),
            (9.223371e18, 9223370937343148032),
            (9.223372e18, MIN),
        ];
        for i in 0..inputs.len() {
            let (xi, e) = inputs[i];
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvttss_si64(x);
            assert_eq!(
                e, r,
                "TestCase #{} _mm_cvttss_si64({:?}) = {}, expected: {}",
                i, x, r, e
            );
        }
    }

    #[simd_test = "sse"]
    pub unsafe fn test_mm_cvtsi64_ss() {
        let inputs = &[
            (4555i64, 4555.0f32),
            (322223333, 322223330.0),
            (-432, -432.0),
            (-322223333, -322223330.0),
            (9223372036854775807, 9.223372e18),
            (-9223372036854775808, -9.223372e18),
        ];

        for i in 0..inputs.len() {
            let (x, f) = inputs[i];
            let a = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
            let r = _mm_cvtsi64_ss(a, x);
            let e = _mm_setr_ps(f, 6.0, 7.0, 8.0);
            assert_eq_m128(e, r);
        }
    }
}
