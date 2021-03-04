use crate::{
    core_arch::{simd::*, x86::*},
    mem::transmute,
};

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sha1msg1"]
    fn sha1msg1(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sha1msg2"]
    fn sha1msg2(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sha1nexte"]
    fn sha1nexte(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sha1rnds4"]
    fn sha1rnds4(a: i32x4, b: i32x4, c: i8) -> i32x4;
    #[link_name = "llvm.x86.sha256msg1"]
    fn sha256msg1(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sha256msg2"]
    fn sha256msg2(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sha256rnds2"]
    fn sha256rnds2(a: i32x4, b: i32x4, k: i32x4) -> i32x4;
}

#[cfg(test)]
use stdarch_test::assert_instr;

/// Performs an intermediate calculation for the next four SHA1 message values
/// (unsigned 32-bit integers) using previous message values from `a` and `b`,
/// and returning the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha1msg1_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha1msg1))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha1msg1_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(sha1msg1(a.as_i32x4(), b.as_i32x4()))
}

/// Performs the final calculation for the next four SHA1 message values
/// (unsigned 32-bit integers) using the intermediate result in `a` and the
/// previous message values in `b`, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha1msg2_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha1msg2))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha1msg2_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(sha1msg2(a.as_i32x4(), b.as_i32x4()))
}

/// Calculate SHA1 state variable E after four rounds of operation from the
/// current SHA1 state variable `a`, add that value to the scheduled values
/// (unsigned 32-bit integers) in `b`, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha1nexte_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha1nexte))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha1nexte_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(sha1nexte(a.as_i32x4(), b.as_i32x4()))
}

/// Performs four rounds of SHA1 operation using an initial SHA1 state (A,B,C,D)
/// from `a` and some pre-computed sum of the next 4 round message values
/// (unsigned 32-bit integers), and state variable E from `b`, and return the
/// updated SHA1 state (A,B,C,D). `FUNC` contains the logic functions and round
/// constants.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha1rnds4_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha1rnds4, FUNC = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha1rnds4_epu32<const FUNC: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm2!(FUNC);
    transmute(sha1rnds4(a.as_i32x4(), b.as_i32x4(), FUNC as i8))
}

/// Performs an intermediate calculation for the next four SHA256 message values
/// (unsigned 32-bit integers) using previous message values from `a` and `b`,
/// and return the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha256msg1_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha256msg1))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha256msg1_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(sha256msg1(a.as_i32x4(), b.as_i32x4()))
}

/// Performs the final calculation for the next four SHA256 message values
/// (unsigned 32-bit integers) using previous message values from `a` and `b`,
/// and return the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha256msg2_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha256msg2))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha256msg2_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(sha256msg2(a.as_i32x4(), b.as_i32x4()))
}

/// Performs 2 rounds of SHA256 operation using an initial SHA256 state
/// (C,D,G,H) from `a`, an initial SHA256 state (A,B,E,F) from `b`, and a
/// pre-computed sum of the next 2 round message values (unsigned 32-bit
/// integers) and the corresponding round constants from `k`, and store the
/// updated SHA256 state (A,B,E,F) in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sha256rnds2_epu32)
#[inline]
#[target_feature(enable = "sha")]
#[cfg_attr(test, assert_instr(sha256rnds2))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sha256rnds2_epu32(a: __m128i, b: __m128i, k: __m128i) -> __m128i {
    transmute(sha256rnds2(a.as_i32x4(), b.as_i32x4(), k.as_i32x4()))
}

#[cfg(test)]
mod tests {
    use std::{
        f32,
        f64::{self, NAN},
        i32,
        mem::{self, transmute},
    };

    use crate::{
        core_arch::{simd::*, x86::*},
        hint::black_box,
    };
    use stdarch_test::simd_test;

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha1msg1_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0x98829f34f74ad457, 0xda2b1a44d0b5ad3c);
        let r = _mm_sha1msg1_epu32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha1msg2_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0xf714b202d863d47d, 0x90c30d946b3d3b35);
        let r = _mm_sha1msg2_epu32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha1nexte_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0x2589d5be923f82a4, 0x59f111f13956c25b);
        let r = _mm_sha1nexte_epu32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha1rnds4_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0x32b13cd8322f5268, 0xc54420862bd9246f);
        let r = _mm_sha1rnds4_epu32::<0>(a, b);
        assert_eq_m128i(r, expected);

        let expected = _mm_set_epi64x(0x6d4c43e56a3c25d9, 0xa7e00fb775cbd3fe);
        let r = _mm_sha1rnds4_epu32::<1>(a, b);
        assert_eq_m128i(r, expected);

        let expected = _mm_set_epi64x(0xb304e383c01222f4, 0x66f6b3b1f89d8001);
        let r = _mm_sha1rnds4_epu32::<2>(a, b);
        assert_eq_m128i(r, expected);

        let expected = _mm_set_epi64x(0x8189b758bfabfa79, 0xdb08f6e78cae098b);
        let r = _mm_sha1rnds4_epu32::<3>(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha256msg1_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0xeb84973fd5cda67d, 0x2857b88f406b09ee);
        let r = _mm_sha256msg1_epu32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha256msg2_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let expected = _mm_set_epi64x(0xb58777ce887fd851, 0x15d1ec8b73ac8450);
        let r = _mm_sha256msg2_epu32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "sha")]
    #[allow(overflowing_literals)]
    unsafe fn test_mm_sha256rnds2_epu32() {
        let a = _mm_set_epi64x(0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        let b = _mm_set_epi64x(0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        let k = _mm_set_epi64x(0, 0x12835b01d807aa98);
        let expected = _mm_set_epi64x(0xd3063037effb15ea, 0x187ee3db0d6d1d19);
        let r = _mm_sha256rnds2_epu32(a, b, k);
        assert_eq_m128i(r, expected);
    }
}
