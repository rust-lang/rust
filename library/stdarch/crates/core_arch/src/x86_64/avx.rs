//! Advanced Vector Extensions (AVX)
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//! Instruction Set Reference, A-Z][intel64_ref]. - [AMD64 Architecture
//! Programmer's Manual, Volume 3: General-Purpose and System
//! Instructions][amd64_ref].
//!
//! [Wikipedia][wiki] provides a quick overview of the instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions

use crate::{
    core_arch::{simd_llvm::*, x86::*},
    mem::transmute,
};

/// Copies `a` to result, and insert the 64-bit integer `i` into result
/// at the location specified by `index`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insert_epi64)
#[inline]
#[rustc_legacy_const_generics(2)]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insert_epi64<const INDEX: i32>(a: __m256i, i: i64) -> __m256i {
    static_assert_imm2!(INDEX);
    transmute(simd_insert(a.as_i64x4(), INDEX as u32, i))
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insert_epi64() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_insert_epi64::<3>(a, 0);
        let e = _mm256_setr_epi64x(1, 2, 3, 0);
        assert_eq_m256i(r, e);
    }
}
