//! Advanced Vector Extensions 2 (AVX)
//!
//! AVX2 expands most AVX commands to 256-bit wide vector registers and
//! adds [FMA](https://en.wikipedia.org/wiki/Fused_multiply-accumulate).
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions][amd64_ref].
//!
//! Wikipedia's [AVX][wiki_avx] and [FMA][wiki_fma] pages provide a quick
//! overview of the instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki_avx]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
//! [wiki_fma]: https://en.wikipedia.org/wiki/Fused_multiply-accumulate

use crate::core_arch::{simd_llvm::*, x86::*};

/// Extracts a 64-bit integer from `a`, selected with `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extract_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[rustc_args_required_const(1)]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extract_epi64(a: __m256i, imm8: i32) -> i64 {
    let a = a.as_i64x4();
    match imm8 & 3 {
        0 => simd_extract(a, 0),
        1 => simd_extract(a, 1),
        2 => simd_extract(a, 2),
        _ => simd_extract(a, 3),
    }
}

#[cfg(test)]
mod tests {
    use crate::core_arch::arch::x86_64::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_extract_epi64() {
        let a = _mm256_setr_epi64x(0, 1, 2, 3);
        let r = _mm256_extract_epi64(a, 3);
        assert_eq!(r, 3);
    }
}
