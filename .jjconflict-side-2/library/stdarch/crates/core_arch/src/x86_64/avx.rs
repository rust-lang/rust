//! Advanced Vector Extensions (AVX)
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref]. - [AMD64 Architecture
//!   Programmer's Manual, Volume 3: General-Purpose and System
//!   Instructions][amd64_ref].
//!
//! [Wikipedia][wiki] provides a quick overview of the instructions available.
//!
//! [intel64_ref]: https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: https://docs.amd.com/v/u/en-US/24594_3.37
//! [wiki]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions

use crate::{core_arch::x86::*, mem::transmute};

/// Copies `a` to result, and insert the 64-bit integer `i` into result
/// at the location specified by `index`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_insert_epi64)
#[inline]
#[rustc_legacy_const_generics(2)]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_const_unstable(feature = "stdarch_const_x86", issue = "149298")]
pub const fn _mm256_insert_epi64<const INDEX: i32>(a: __m256i, i: i64) -> __m256i {
    static_assert_uimm_bits!(INDEX, 2);
    unsafe { transmute(simd_insert!(a.as_i64x4(), INDEX as u32, i)) }
}

/// Extracts a 64-bit integer from `a`, selected with `INDEX`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extract_epi64)
#[inline]
#[target_feature(enable = "avx")]
#[rustc_legacy_const_generics(1)]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_const_unstable(feature = "stdarch_const_x86", issue = "149298")]
pub const fn _mm256_extract_epi64<const INDEX: i32>(a: __m256i) -> i64 {
    static_assert_uimm_bits!(INDEX, 2);
    unsafe { simd_extract!(a.as_i64x4(), INDEX as u32) }
}

#[cfg(test)]
mod tests {
    use crate::core_arch::assert_eq_const as assert_eq;
    use stdarch_test::simd_test;

    use crate::core_arch::arch::x86_64::*;

    #[simd_test(enable = "avx")]
    const fn test_mm256_insert_epi64() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_insert_epi64::<3>(a, 0);
        let e = _mm256_setr_epi64x(1, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    const fn test_mm256_extract_epi64() {
        let a = _mm256_setr_epi64x(0, 1, 2, 3);
        let r = _mm256_extract_epi64::<3>(a);
        assert_eq!(r, 3);
    }
}
