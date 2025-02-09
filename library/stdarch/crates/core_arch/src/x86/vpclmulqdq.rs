//! Vectorized Carry-less Multiplication (VCLMUL)
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref] (p. 4-241).
//!
//! [intel64_ref]: http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::x86::__m256i;
use crate::core_arch::x86::__m512i;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.pclmulqdq.256"]
    fn pclmulqdq_256(a: __m256i, round_key: __m256i, imm8: u8) -> __m256i;
    #[link_name = "llvm.x86.pclmulqdq.512"]
    fn pclmulqdq_512(a: __m512i, round_key: __m512i, imm8: u8) -> __m512i;
}

// for some odd reason on x86_64 we generate the correct long name instructions
// but on i686 we generate the short name + imm8
// so we need to special-case on that...

/// Performs a carry-less multiplication of two 64-bit polynomials over the
/// finite field GF(2) - in each of the 4 128-bit lanes.
///
/// The immediate byte is used for determining which halves of each lane `a` and `b`
/// should be used. Immediate bits other than 0 and 4 are ignored.
/// All lanes share immediate byte.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_clmulepi64_epi128)
#[inline]
#[target_feature(enable = "vpclmulqdq,avx512f")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
// technically according to Intel's documentation we don't need avx512f here, however LLVM gets confused otherwise
#[cfg_attr(test, assert_instr(vpclmul, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn _mm512_clmulepi64_epi128<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    pclmulqdq_512(a, b, IMM8 as u8)
}

/// Performs a carry-less multiplication of two 64-bit polynomials over the
/// finite field GF(2) - in each of the 2 128-bit lanes.
///
/// The immediate byte is used for determining which halves of each lane `a` and `b`
/// should be used. Immediate bits other than 0 and 4 are ignored.
/// All lanes share immediate byte.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_clmulepi64_epi128)
#[inline]
#[target_feature(enable = "vpclmulqdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpclmul, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn _mm256_clmulepi64_epi128<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    pclmulqdq_256(a, b, IMM8 as u8)
}

#[cfg(test)]
mod tests {
    // The constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __mXXXi happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    macro_rules! verify_kat_pclmul {
        ($broadcast:ident, $clmul:ident, $assert:ident) => {
            // Constants taken from https://software.intel.com/sites/default/files/managed/72/cc/clmul-wp-rev-2.02-2014-04-20.pdf
         let a = _mm_set_epi64x(0x7b5b546573745665, 0x63746f725d53475d);
         let a = $broadcast(a);
         let b = _mm_set_epi64x(0x4869285368617929, 0x5b477565726f6e5d);
         let b = $broadcast(b);
         let r00 = _mm_set_epi64x(0x1d4d84c85c3440c0, 0x929633d5d36f0451);
         let r00 = $broadcast(r00);
         let r01 = _mm_set_epi64x(0x1bd17c8d556ab5a1, 0x7fa540ac2a281315);
         let r01 = $broadcast(r01);
         let r10 = _mm_set_epi64x(0x1a2bf6db3a30862f, 0xbabf262df4b7d5c9);
         let r10 = $broadcast(r10);
         let r11 = _mm_set_epi64x(0x1d1e1f2c592e7c45, 0xd66ee03e410fd4ed);
         let r11 = $broadcast(r11);

         $assert($clmul::<0x00>(a, b), r00);
         $assert($clmul::<0x10>(a, b), r01);
         $assert($clmul::<0x01>(a, b), r10);
         $assert($clmul::<0x11>(a, b), r11);

         let a0 = _mm_set_epi64x(0x0000000000000000, 0x8000000000000000);
         let a0 = $broadcast(a0);
         let r = _mm_set_epi64x(0x4000000000000000, 0x0000000000000000);
         let r = $broadcast(r);
         $assert($clmul::<0x00>(a0, a0), r);
        }
    }

    macro_rules! unroll {
        ($target:ident[4] = $op:ident::<4>($source:ident);) => {
            $target[3] = $op::<3>($source);
            $target[2] = $op::<2>($source);
            unroll! {$target[2] = $op::<2>($source);}
        };
        ($target:ident[2] = $op:ident::<2>($source:ident);) => {
            $target[1] = $op::<1>($source);
            $target[0] = $op::<0>($source);
        };
        (assert_eq_m128i($op:ident::<4>($vec_res:ident),$lin_res:ident[4]);) => {
            assert_eq_m128i($op::<3>($vec_res), $lin_res[3]);
            assert_eq_m128i($op::<2>($vec_res), $lin_res[2]);
            unroll! {assert_eq_m128i($op::<2>($vec_res),$lin_res[2]);}
        };
        (assert_eq_m128i($op:ident::<2>($vec_res:ident),$lin_res:ident[2]);) => {
            assert_eq_m128i($op::<1>($vec_res), $lin_res[1]);
            assert_eq_m128i($op::<0>($vec_res), $lin_res[0]);
        };
    }

    // this function tests one of the possible 4 instances
    // with different inputs across lanes
    #[target_feature(enable = "vpclmulqdq,avx512f")]
    unsafe fn verify_512_helper(
        linear: unsafe fn(__m128i, __m128i) -> __m128i,
        vectorized: unsafe fn(__m512i, __m512i) -> __m512i,
    ) {
        let a = _mm512_set_epi64(
            0xDCB4DB3657BF0B7D,
            0x18DB0601068EDD9F,
            0xB76B908233200DC5,
            0xE478235FA8E22D5E,
            0xAB05CFFA2621154C,
            0x1171B47A186174C9,
            0x8C6B6C0E7595CEC9,
            0xBE3E7D4934E961BD,
        );
        let b = _mm512_set_epi64(
            0x672F6F105A94CEA7,
            0x8298B8FFCA5F829C,
            0xA3927047B3FB61D8,
            0x978093862CDE7187,
            0xB1927AB22F31D0EC,
            0xA9A5DA619BE4D7AF,
            0xCA2590F56884FDC6,
            0x19BE9F660038BDB5,
        );

        let mut a_decomp = [_mm_setzero_si128(); 4];
        unroll! {a_decomp[4] = _mm512_extracti32x4_epi32::<4>(a);}
        let mut b_decomp = [_mm_setzero_si128(); 4];
        unroll! {b_decomp[4] = _mm512_extracti32x4_epi32::<4>(b);}

        let r = vectorized(a, b);
        let mut e_decomp = [_mm_setzero_si128(); 4];
        for i in 0..4 {
            e_decomp[i] = linear(a_decomp[i], b_decomp[i]);
        }
        unroll! {assert_eq_m128i(_mm512_extracti32x4_epi32::<4>(r),e_decomp[4]);}
    }

    // this function tests one of the possible 4 instances
    // with different inputs across lanes for the VL version
    #[target_feature(enable = "vpclmulqdq,avx512vl")]
    unsafe fn verify_256_helper(
        linear: unsafe fn(__m128i, __m128i) -> __m128i,
        vectorized: unsafe fn(__m256i, __m256i) -> __m256i,
    ) {
        let a = _mm512_set_epi64(
            0xDCB4DB3657BF0B7D,
            0x18DB0601068EDD9F,
            0xB76B908233200DC5,
            0xE478235FA8E22D5E,
            0xAB05CFFA2621154C,
            0x1171B47A186174C9,
            0x8C6B6C0E7595CEC9,
            0xBE3E7D4934E961BD,
        );
        let b = _mm512_set_epi64(
            0x672F6F105A94CEA7,
            0x8298B8FFCA5F829C,
            0xA3927047B3FB61D8,
            0x978093862CDE7187,
            0xB1927AB22F31D0EC,
            0xA9A5DA619BE4D7AF,
            0xCA2590F56884FDC6,
            0x19BE9F660038BDB5,
        );

        let mut a_decomp = [_mm_setzero_si128(); 2];
        unroll! {a_decomp[2] = _mm512_extracti32x4_epi32::<2>(a);}
        let mut b_decomp = [_mm_setzero_si128(); 2];
        unroll! {b_decomp[2] = _mm512_extracti32x4_epi32::<2>(b);}

        let r = vectorized(
            _mm512_extracti64x4_epi64::<0>(a),
            _mm512_extracti64x4_epi64::<0>(b),
        );
        let mut e_decomp = [_mm_setzero_si128(); 2];
        for i in 0..2 {
            e_decomp[i] = linear(a_decomp[i], b_decomp[i]);
        }
        unroll! {assert_eq_m128i(_mm256_extracti128_si256::<2>(r),e_decomp[2]);}
    }

    #[simd_test(enable = "vpclmulqdq,avx512f")]
    unsafe fn test_mm512_clmulepi64_epi128() {
        verify_kat_pclmul!(
            _mm512_broadcast_i32x4,
            _mm512_clmulepi64_epi128,
            assert_eq_m512i
        );

        verify_512_helper(
            |a, b| _mm_clmulepi64_si128::<0x00>(a, b),
            |a, b| _mm512_clmulepi64_epi128::<0x00>(a, b),
        );
        verify_512_helper(
            |a, b| _mm_clmulepi64_si128::<0x01>(a, b),
            |a, b| _mm512_clmulepi64_epi128::<0x01>(a, b),
        );
        verify_512_helper(
            |a, b| _mm_clmulepi64_si128::<0x10>(a, b),
            |a, b| _mm512_clmulepi64_epi128::<0x10>(a, b),
        );
        verify_512_helper(
            |a, b| _mm_clmulepi64_si128::<0x11>(a, b),
            |a, b| _mm512_clmulepi64_epi128::<0x11>(a, b),
        );
    }

    #[simd_test(enable = "vpclmulqdq,avx512vl")]
    unsafe fn test_mm256_clmulepi64_epi128() {
        verify_kat_pclmul!(
            _mm256_broadcastsi128_si256,
            _mm256_clmulepi64_epi128,
            assert_eq_m256i
        );

        verify_256_helper(
            |a, b| _mm_clmulepi64_si128::<0x00>(a, b),
            |a, b| _mm256_clmulepi64_epi128::<0x00>(a, b),
        );
        verify_256_helper(
            |a, b| _mm_clmulepi64_si128::<0x01>(a, b),
            |a, b| _mm256_clmulepi64_epi128::<0x01>(a, b),
        );
        verify_256_helper(
            |a, b| _mm_clmulepi64_si128::<0x10>(a, b),
            |a, b| _mm256_clmulepi64_epi128::<0x10>(a, b),
        );
        verify_256_helper(
            |a, b| _mm_clmulepi64_si128::<0x11>(a, b),
            |a, b| _mm256_clmulepi64_epi128::<0x11>(a, b),
        );
    }
}
