//! Galois Field New Instructions (GFNI)
//!
//! The intrinsics here correspond to those in the `immintrin.h` C header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::simd::i8x16;
use crate::core_arch::simd::i8x32;
use crate::core_arch::simd::i8x64;
use crate::core_arch::simd_llvm::simd_select_bitmask;
use crate::core_arch::x86::__m128i;
use crate::core_arch::x86::__m256i;
use crate::core_arch::x86::__m512i;
use crate::core_arch::x86::__mmask16;
use crate::core_arch::x86::__mmask32;
use crate::core_arch::x86::__mmask64;
use crate::core_arch::x86::_mm256_setzero_si256;
use crate::core_arch::x86::_mm512_setzero_si512;
use crate::core_arch::x86::_mm_setzero_si128;
use crate::core_arch::x86::m128iExt;
use crate::core_arch::x86::m256iExt;
use crate::core_arch::x86::m512iExt;
use crate::mem::transmute;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.vgf2p8affineinvqb.512"]
    fn vgf2p8affineinvqb_512(x: i8x64, a: i8x64, imm8: u8) -> i8x64;
    #[link_name = "llvm.x86.vgf2p8affineinvqb.256"]
    fn vgf2p8affineinvqb_256(x: i8x32, a: i8x32, imm8: u8) -> i8x32;
    #[link_name = "llvm.x86.vgf2p8affineinvqb.128"]
    fn vgf2p8affineinvqb_128(x: i8x16, a: i8x16, imm8: u8) -> i8x16;
    #[link_name = "llvm.x86.vgf2p8affineqb.512"]
    fn vgf2p8affineqb_512(x: i8x64, a: i8x64, imm8: u8) -> i8x64;
    #[link_name = "llvm.x86.vgf2p8affineqb.256"]
    fn vgf2p8affineqb_256(x: i8x32, a: i8x32, imm8: u8) -> i8x32;
    #[link_name = "llvm.x86.vgf2p8affineqb.128"]
    fn vgf2p8affineqb_128(x: i8x16, a: i8x16, imm8: u8) -> i8x16;
    #[link_name = "llvm.x86.vgf2p8mulb.512"]
    fn vgf2p8mulb_512(a: i8x64, b: i8x64) -> i8x64;
    #[link_name = "llvm.x86.vgf2p8mulb.256"]
    fn vgf2p8mulb_256(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.vgf2p8mulb.128"]
    fn vgf2p8mulb_128(a: i8x16, b: i8x16) -> i8x16;
}

// LLVM requires AVX512BW for a lot of these instructions, see
// https://github.com/llvm/llvm-project/blob/release/9.x/clang/include/clang/Basic/BuiltinsX86.def#L457
// however our tests also require the target feature list to match Intel's
// which *doesn't* require AVX512BW but only AVX512F, so we added the redundant AVX512F
// requirement (for now)
// also see
// https://github.com/llvm/llvm-project/blob/release/9.x/clang/lib/Headers/gfniintrin.h
// for forcing GFNI, BW and optionally VL extension

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm512_gf2p8mul_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vgf2p8mulb_512(a.as_i8x64(), b.as_i8x64()))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm512_mask_gf2p8mul_epi8(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_512(a.as_i8x64(), b.as_i8x64()),
        src.as_i8x64(),
    ))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm512_maskz_gf2p8mul_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_512(a.as_i8x64(), b.as_i8x64()),
        zero,
    ))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm256_gf2p8mul_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(vgf2p8mulb_256(a.as_i8x32(), b.as_i8x32()))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm256_mask_gf2p8mul_epi8(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_256(a.as_i8x32(), b.as_i8x32()),
        src.as_i8x32(),
    ))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm256_maskz_gf2p8mul_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_256(a.as_i8x32(), b.as_i8x32()),
        zero,
    ))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm_gf2p8mul_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(vgf2p8mulb_128(a.as_i8x16(), b.as_i8x16()))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm_mask_gf2p8mul_epi8(
    src: __m128i,
    k: __mmask16,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_128(a.as_i8x16(), b.as_i8x16()),
        src.as_i8x16(),
    ))
}

/// Performs a multiplication in GF(2^8) on the packed bytes.
/// The field is in polynomial representation with the reduction polynomial
///  x^8 + x^4 + x^3 + x + 1.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_gf2p8mul_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8mulb))]
pub unsafe fn _mm_maskz_gf2p8mul_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(
        k,
        vgf2p8mulb_128(a.as_i8x16(), b.as_i8x16()),
        zero,
    ))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_gf2p8affine_epi64_epi8(x: __m512i, a: __m512i, b: i32) -> __m512i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_gf2p8affine_epi64_epi8(
    k: __mmask64,
    x: __m512i,
    a: __m512i,
    b: i32,
) -> __m512i {
    let zero = _mm512_setzero_si512().as_i8x64();
    assert!(0 <= b && b < 256);
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_gf2p8affine_epi64_epi8(
    src: __m512i,
    k: __mmask64,
    x: __m512i,
    a: __m512i,
    b: i32,
) -> __m512i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x64()))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_gf2p8affine_epi64_epi8(x: __m256i, a: __m256i, b: i32) -> __m256i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_maskz_gf2p8affine_epi64_epi8(
    k: __mmask32,
    x: __m256i,
    a: __m256i,
    b: i32,
) -> __m256i {
    let zero = _mm256_setzero_si256().as_i8x32();
    assert!(0 <= b && b < 256);
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm256_mask_gf2p8affine_epi64_epi8(
    src: __m256i,
    k: __mmask32,
    x: __m256i,
    a: __m256i,
    b: i32,
) -> __m256i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x32()))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_gf2p8affine_epi64_epi8(x: __m128i, a: __m128i, b: i32) -> __m128i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_maskz_gf2p8affine_epi64_epi8(
    k: __mmask16,
    x: __m128i,
    a: __m128i,
    b: i32,
) -> __m128i {
    let zero = _mm_setzero_si128().as_i8x16();
    assert!(0 <= b && b < 256);
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the packed bytes in x.
/// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_gf2p8affine_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm_mask_gf2p8affine_epi64_epi8(
    src: __m128i,
    k: __mmask16,
    x: __m128i,
    a: __m128i,
    b: i32,
) -> __m128i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x16()))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_gf2p8affineinv_epi64_epi8(x: __m512i, a: __m512i, b: i32) -> __m512i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_gf2p8affineinv_epi64_epi8(
    k: __mmask64,
    x: __m512i,
    a: __m512i,
    b: i32,
) -> __m512i {
    assert!(0 <= b && b < 256);
    let zero = _mm512_setzero_si512().as_i8x64();
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512f")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_gf2p8affineinv_epi64_epi8(
    src: __m512i,
    k: __mmask64,
    x: __m512i,
    a: __m512i,
    b: i32,
) -> __m512i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x64();
    let a = a.as_i8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_512(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x64()))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_gf2p8affineinv_epi64_epi8(x: __m256i, a: __m256i, b: i32) -> __m256i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_maskz_gf2p8affineinv_epi64_epi8(
    k: __mmask32,
    x: __m256i,
    a: __m256i,
    b: i32,
) -> __m256i {
    assert!(0 <= b && b < 256);
    let zero = _mm256_setzero_si256().as_i8x32();
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm256_mask_gf2p8affineinv_epi64_epi8(
    src: __m256i,
    k: __mmask32,
    x: __m256i,
    a: __m256i,
    b: i32,
) -> __m256i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x32();
    let a = a.as_i8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_256(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x32()))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_gf2p8affineinv_epi64_epi8(x: __m128i, a: __m128i, b: i32) -> __m128i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(r)
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_maskz_gf2p8affineinv_epi64_epi8(
    k: __mmask16,
    x: __m128i,
    a: __m128i,
    b: i32,
) -> __m128i {
    assert!(0 <= b && b < 256);
    let zero = _mm_setzero_si128().as_i8x16();
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, zero))
}

/// Performs an affine transformation on the inverted packed bytes in x.
/// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit matrix
/// and b being a constant 8-bit immediate value.
/// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
/// The inverse of 0 is 0.
/// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_gf2p8affineinv_epi64_epi8)
#[inline]
#[target_feature(enable = "avx512gfni,avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vgf2p8affineinvqb, b = 0))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm_mask_gf2p8affineinv_epi64_epi8(
    src: __m128i,
    k: __mmask16,
    x: __m128i,
    a: __m128i,
    b: i32,
) -> __m128i {
    assert!(0 <= b && b < 256);
    let x = x.as_i8x16();
    let a = a.as_i8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgf2p8affineinvqb_128(x, a, $imm8)
        };
    }
    let r = constify_imm8_sae!(b, call);
    transmute(simd_select_bitmask(k, r, src.as_i8x16()))
}

#[cfg(test)]
mod tests {
    // The constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __mXXXi happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use core::hint::black_box;
    use core::intrinsics::size_of;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    fn mulbyte(left: u8, right: u8) -> u8 {
        // this implementation follows the description in
        // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_gf2p8mul_epi8
        const REDUCTION_POLYNOMIAL: u16 = 0x11b;
        let left: u16 = left.into();
        let right: u16 = right.into();
        let mut carryless_product: u16 = 0;

        // Carryless multiplication
        for i in 0..8 {
            if ((left >> i) & 0x01) != 0 {
                carryless_product ^= right << i;
            }
        }

        // reduction, adding in "0" where appropriate to clear out high bits
        // note that REDUCTION_POLYNOMIAL is zero in this context
        for i in (8..=14).rev() {
            if ((carryless_product >> i) & 0x01) != 0 {
                carryless_product ^= REDUCTION_POLYNOMIAL << (i - 8);
            }
        }

        carryless_product as u8
    }

    const NUM_TEST_WORDS_512: usize = 4;
    const NUM_TEST_WORDS_256: usize = NUM_TEST_WORDS_512 * 2;
    const NUM_TEST_WORDS_128: usize = NUM_TEST_WORDS_256 * 2;
    const NUM_TEST_ENTRIES: usize = NUM_TEST_WORDS_512 * 64;
    const NUM_TEST_WORDS_64: usize = NUM_TEST_WORDS_128 * 2;
    const NUM_BYTES: usize = 256;
    const NUM_BYTES_WORDS_128: usize = NUM_BYTES / 16;
    const NUM_BYTES_WORDS_256: usize = NUM_BYTES_WORDS_128 / 2;
    const NUM_BYTES_WORDS_512: usize = NUM_BYTES_WORDS_256 / 2;

    fn parity(input: u8) -> u8 {
        let mut accumulator = 0;
        for i in 0..8 {
            accumulator ^= (input >> i) & 0x01;
        }
        accumulator
    }

    fn mat_vec_multiply_affine(matrix: u64, x: u8, b: u8) -> u8 {
        // this implementation follows the description in
        // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_gf2p8affine_epi64_epi8
        let mut accumulator = 0;

        for bit in 0..8 {
            accumulator |= parity(x & matrix.to_le_bytes()[bit]) << (7 - bit);
        }

        accumulator ^ b
    }

    fn generate_affine_mul_test_data(
        immediate: u8,
    ) -> (
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ) {
        let mut left: [u64; NUM_TEST_WORDS_64] = [0; NUM_TEST_WORDS_64];
        let mut right: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
        let mut result: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];

        for i in 0..NUM_TEST_WORDS_64 {
            left[i] = (i as u64) * 103 * 101;
            for j in 0..8 {
                let j64 = j as u64;
                right[i * 8 + j] = ((left[i] + j64) % 256) as u8;
                result[i * 8 + j] = mat_vec_multiply_affine(left[i], right[i * 8 + j], immediate);
            }
        }

        (left, right, result)
    }

    fn generate_inv_tests_data() -> ([u8; NUM_BYTES], [u8; NUM_BYTES]) {
        let mut input: [u8; NUM_BYTES] = [0; NUM_BYTES];
        let mut result: [u8; NUM_BYTES] = [0; NUM_BYTES];

        for i in 0..NUM_BYTES {
            input[i] = (i % 256) as u8;
            result[i] = if i == 0 { 0 } else { 1 };
        }

        (input, result)
    }

    const AES_S_BOX: [u8; NUM_BYTES] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
        0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
        0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
        0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
        0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
        0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
        0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
        0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
        0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
        0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
        0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
        0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
        0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
        0x16,
    ];

    fn generate_byte_mul_test_data() -> (
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ) {
        let mut left: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
        let mut right: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
        let mut result: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];

        for i in 0..NUM_TEST_ENTRIES {
            left[i] = (i % 256) as u8;
            right[i] = left[i] * 101;
            result[i] = mulbyte(left[i], right[i]);
        }

        (left, right, result)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn load_m128i_word<T>(data: &[T], word_index: usize) -> __m128i {
        let byte_offset = word_index * 16 / size_of::<T>();
        let pointer = data.as_ptr().offset(byte_offset as isize) as *const __m128i;
        _mm_loadu_si128(black_box(pointer))
    }

    #[target_feature(enable = "avx")]
    unsafe fn load_m256i_word<T>(data: &[T], word_index: usize) -> __m256i {
        let byte_offset = word_index * 32 / size_of::<T>();
        let pointer = data.as_ptr().offset(byte_offset as isize) as *const __m256i;
        _mm256_loadu_si256(black_box(pointer))
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn load_m512i_word<T>(data: &[T], word_index: usize) -> __m512i {
        let byte_offset = word_index * 64 / size_of::<T>();
        let pointer = data.as_ptr().offset(byte_offset as isize) as *const i32;
        _mm512_loadu_si512(black_box(pointer))
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_gf2p8mul_epi8() {
        let (left, right, expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_512 {
            let left = load_m512i_word(&left, i);
            let right = load_m512i_word(&right, i);
            let expected = load_m512i_word(&expected, i);
            let result = _mm512_gf2p8mul_epi8(left, right);
            assert_eq_m512i(result, expected);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_maskz_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_512 {
            let left = load_m512i_word(&left, i);
            let right = load_m512i_word(&right, i);
            let result_zero = _mm512_maskz_gf2p8mul_epi8(0, left, right);
            assert_eq_m512i(result_zero, _mm512_setzero_si512());
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8mul_epi8(left, right);
            let result_masked = _mm512_maskz_gf2p8mul_epi8(mask_bytes, left, right);
            let expected_masked =
                _mm512_mask_blend_epi32(mask_words, _mm512_setzero_si512(), expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_mask_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_512 {
            let left = load_m512i_word(&left, i);
            let right = load_m512i_word(&right, i);
            let result_left = _mm512_mask_gf2p8mul_epi8(left, 0, left, right);
            assert_eq_m512i(result_left, left);
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8mul_epi8(left, right);
            let result_masked = _mm512_mask_gf2p8mul_epi8(left, mask_bytes, left, right);
            let expected_masked = _mm512_mask_blend_epi32(mask_words, left, expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_gf2p8mul_epi8() {
        let (left, right, expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_256 {
            let left = load_m256i_word(&left, i);
            let right = load_m256i_word(&right, i);
            let expected = load_m256i_word(&expected, i);
            let result = _mm256_gf2p8mul_epi8(left, right);
            assert_eq_m256i(result, expected);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_256 {
            let left = load_m256i_word(&left, i);
            let right = load_m256i_word(&right, i);
            let result_zero = _mm256_maskz_gf2p8mul_epi8(0, left, right);
            assert_eq_m256i(result_zero, _mm256_setzero_si256());
            let mask_bytes: __mmask32 = 0x0F_F0_FF_00;
            const MASK_WORDS: i32 = 0b01_10_11_00;
            let expected_result = _mm256_gf2p8mul_epi8(left, right);
            let result_masked = _mm256_maskz_gf2p8mul_epi8(mask_bytes, left, right);
            let expected_masked =
                _mm256_blend_epi32(_mm256_setzero_si256(), expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_256 {
            let left = load_m256i_word(&left, i);
            let right = load_m256i_word(&right, i);
            let result_left = _mm256_mask_gf2p8mul_epi8(left, 0, left, right);
            assert_eq_m256i(result_left, left);
            let mask_bytes: __mmask32 = 0x0F_F0_FF_00;
            const MASK_WORDS: i32 = 0b01_10_11_00;
            let expected_result = _mm256_gf2p8mul_epi8(left, right);
            let result_masked = _mm256_mask_gf2p8mul_epi8(left, mask_bytes, left, right);
            let expected_masked = _mm256_blend_epi32(left, expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_gf2p8mul_epi8() {
        let (left, right, expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_128 {
            let left = load_m128i_word(&left, i);
            let right = load_m128i_word(&right, i);
            let expected = load_m128i_word(&expected, i);
            let result = _mm_gf2p8mul_epi8(left, right);
            assert_eq_m128i(result, expected);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_128 {
            let left = load_m128i_word(&left, i);
            let right = load_m128i_word(&right, i);
            let result_zero = _mm_maskz_gf2p8mul_epi8(0, left, right);
            assert_eq_m128i(result_zero, _mm_setzero_si128());
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8mul_epi8(left, right);
            let result_masked = _mm_maskz_gf2p8mul_epi8(mask_bytes, left, right);
            let expected_masked =
                _mm_blend_epi32::<MASK_WORDS>(_mm_setzero_si128(), expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_gf2p8mul_epi8() {
        let (left, right, _expected) = generate_byte_mul_test_data();

        for i in 0..NUM_TEST_WORDS_128 {
            let left = load_m128i_word(&left, i);
            let right = load_m128i_word(&right, i);
            let result_left = _mm_mask_gf2p8mul_epi8(left, 0, left, right);
            assert_eq_m128i(result_left, left);
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8mul_epi8(left, right);
            let result_masked = _mm_mask_gf2p8mul_epi8(left, mask_bytes, left, right);
            let expected_masked = _mm_blend_epi32::<MASK_WORDS>(left, expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_gf2p8affine_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        let constant: i64 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm512_set1_epi64(identity);
        let constant = _mm512_set1_epi64(constant);
        let constant_reference = _mm512_set1_epi8(CONSTANT_BYTE as i8);

        let (bytes, more_bytes, _) = generate_byte_mul_test_data();
        let (matrices, vectors, references) = generate_affine_mul_test_data(IDENTITY_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let data = load_m512i_word(&bytes, i);
            let result = _mm512_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m512i(result, data);
            let result = _mm512_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m512i(result, constant_reference);
            let data = load_m512i_word(&more_bytes, i);
            let result = _mm512_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m512i(result, data);
            let result = _mm512_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m512i(result, constant_reference);

            let matrix = load_m512i_word(&matrices, i);
            let vector = load_m512i_word(&vectors, i);
            let reference = load_m512i_word(&references, i);

            let result = _mm512_gf2p8affine_epi64_epi8(vector, matrix, IDENTITY_BYTE);
            assert_eq_m512i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_maskz_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let matrix = load_m512i_word(&matrices, i);
            let vector = load_m512i_word(&vectors, i);
            let result_zero = _mm512_maskz_gf2p8affine_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m512i(result_zero, _mm512_setzero_si512());
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8affine_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm512_maskz_gf2p8affine_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm512_mask_blend_epi32(mask_words, _mm512_setzero_si512(), expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_mask_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let left = load_m512i_word(&vectors, i);
            let right = load_m512i_word(&matrices, i);
            let result_left =
                _mm512_mask_gf2p8affine_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m512i(result_left, left);
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8affine_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm512_mask_gf2p8affine_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm512_mask_blend_epi32(mask_words, left, expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_gf2p8affine_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        let constant: i64 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm256_set1_epi64x(identity);
        let constant = _mm256_set1_epi64x(constant);
        let constant_reference = _mm256_set1_epi8(CONSTANT_BYTE as i8);

        let (bytes, more_bytes, _) = generate_byte_mul_test_data();
        let (matrices, vectors, references) = generate_affine_mul_test_data(IDENTITY_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let data = load_m256i_word(&bytes, i);
            let result = _mm256_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m256i(result, data);
            let result = _mm256_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m256i(result, constant_reference);
            let data = load_m256i_word(&more_bytes, i);
            let result = _mm256_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m256i(result, data);
            let result = _mm256_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m256i(result, constant_reference);

            let matrix = load_m256i_word(&matrices, i);
            let vector = load_m256i_word(&vectors, i);
            let reference = load_m256i_word(&references, i);

            let result = _mm256_gf2p8affine_epi64_epi8(vector, matrix, IDENTITY_BYTE);
            assert_eq_m256i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let matrix = load_m256i_word(&matrices, i);
            let vector = load_m256i_word(&vectors, i);
            let result_zero = _mm256_maskz_gf2p8affine_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m256i(result_zero, _mm256_setzero_si256());
            let mask_bytes: __mmask32 = 0xFF_0F_F0_00;
            const MASK_WORDS: i32 = 0b11_01_10_00;
            let expected_result = _mm256_gf2p8affine_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm256_maskz_gf2p8affine_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm256_blend_epi32(_mm256_setzero_si256(), expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let left = load_m256i_word(&vectors, i);
            let right = load_m256i_word(&matrices, i);
            let result_left =
                _mm256_mask_gf2p8affine_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m256i(result_left, left);
            let mask_bytes: __mmask32 = 0xFF_0F_F0_00;
            const MASK_WORDS: i32 = 0b11_01_10_00;
            let expected_result = _mm256_gf2p8affine_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm256_mask_gf2p8affine_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm256_blend_epi32(left, expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_gf2p8affine_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        let constant: i64 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm_set1_epi64x(identity);
        let constant = _mm_set1_epi64x(constant);
        let constant_reference = _mm_set1_epi8(CONSTANT_BYTE as i8);

        let (bytes, more_bytes, _) = generate_byte_mul_test_data();
        let (matrices, vectors, references) = generate_affine_mul_test_data(IDENTITY_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let data = load_m128i_word(&bytes, i);
            let result = _mm_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m128i(result, data);
            let result = _mm_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m128i(result, constant_reference);
            let data = load_m128i_word(&more_bytes, i);
            let result = _mm_gf2p8affine_epi64_epi8(data, identity, IDENTITY_BYTE);
            assert_eq_m128i(result, data);
            let result = _mm_gf2p8affine_epi64_epi8(data, constant, CONSTANT_BYTE);
            assert_eq_m128i(result, constant_reference);

            let matrix = load_m128i_word(&matrices, i);
            let vector = load_m128i_word(&vectors, i);
            let reference = load_m128i_word(&references, i);

            let result = _mm_gf2p8affine_epi64_epi8(vector, matrix, IDENTITY_BYTE);
            assert_eq_m128i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let matrix = load_m128i_word(&matrices, i);
            let vector = load_m128i_word(&vectors, i);
            let result_zero = _mm_maskz_gf2p8affine_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m128i(result_zero, _mm_setzero_si128());
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8affine_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm_maskz_gf2p8affine_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm_blend_epi32::<MASK_WORDS>(_mm_setzero_si128(), expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_gf2p8affine_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let left = load_m128i_word(&vectors, i);
            let right = load_m128i_word(&matrices, i);
            let result_left = _mm_mask_gf2p8affine_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m128i(result_left, left);
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8affine_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm_mask_gf2p8affine_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm_blend_epi32::<MASK_WORDS>(left, expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_gf2p8affineinv_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm512_set1_epi64(identity);

        // validate inversion
        let (inputs, results) = generate_inv_tests_data();

        for i in 0..NUM_BYTES_WORDS_512 {
            let input = load_m512i_word(&inputs, i);
            let reference = load_m512i_word(&results, i);
            let result = _mm512_gf2p8affineinv_epi64_epi8(input, identity, IDENTITY_BYTE);
            let remultiplied = _mm512_gf2p8mul_epi8(result, input);
            assert_eq_m512i(remultiplied, reference);
        }

        // validate subsequent affine operation
        let (matrices, vectors, _affine_expected) =
            generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let vector = load_m512i_word(&vectors, i);
            let matrix = load_m512i_word(&matrices, i);

            let inv_vec = _mm512_gf2p8affineinv_epi64_epi8(vector, identity, IDENTITY_BYTE);
            let reference = _mm512_gf2p8affine_epi64_epi8(inv_vec, matrix, CONSTANT_BYTE);
            let result = _mm512_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            assert_eq_m512i(result, reference);
        }

        // validate everything by virtue of checking against the AES SBox
        const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
        let sbox_matrix = _mm512_set1_epi64(AES_S_BOX_MATRIX);

        for i in 0..NUM_BYTES_WORDS_512 {
            let reference = load_m512i_word(&AES_S_BOX, i);
            let input = load_m512i_word(&inputs, i);
            let result = _mm512_gf2p8affineinv_epi64_epi8(input, sbox_matrix, CONSTANT_BYTE);
            assert_eq_m512i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_maskz_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let matrix = load_m512i_word(&matrices, i);
            let vector = load_m512i_word(&vectors, i);
            let result_zero =
                _mm512_maskz_gf2p8affineinv_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m512i(result_zero, _mm512_setzero_si512());
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm512_maskz_gf2p8affineinv_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm512_mask_blend_epi32(mask_words, _mm512_setzero_si512(), expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw")]
    unsafe fn test_mm512_mask_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_512 {
            let left = load_m512i_word(&vectors, i);
            let right = load_m512i_word(&matrices, i);
            let result_left =
                _mm512_mask_gf2p8affineinv_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m512i(result_left, left);
            let mask_bytes: __mmask64 = 0x0F_0F_0F_0F_FF_FF_00_00;
            let mask_words: __mmask16 = 0b01_01_01_01_11_11_00_00;
            let expected_result = _mm512_gf2p8affineinv_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm512_mask_gf2p8affineinv_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm512_mask_blend_epi32(mask_words, left, expected_result);
            assert_eq_m512i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_gf2p8affineinv_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm256_set1_epi64x(identity);

        // validate inversion
        let (inputs, results) = generate_inv_tests_data();

        for i in 0..NUM_BYTES_WORDS_256 {
            let input = load_m256i_word(&inputs, i);
            let reference = load_m256i_word(&results, i);
            let result = _mm256_gf2p8affineinv_epi64_epi8(input, identity, IDENTITY_BYTE);
            let remultiplied = _mm256_gf2p8mul_epi8(result, input);
            assert_eq_m256i(remultiplied, reference);
        }

        // validate subsequent affine operation
        let (matrices, vectors, _affine_expected) =
            generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let vector = load_m256i_word(&vectors, i);
            let matrix = load_m256i_word(&matrices, i);

            let inv_vec = _mm256_gf2p8affineinv_epi64_epi8(vector, identity, IDENTITY_BYTE);
            let reference = _mm256_gf2p8affine_epi64_epi8(inv_vec, matrix, CONSTANT_BYTE);
            let result = _mm256_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            assert_eq_m256i(result, reference);
        }

        // validate everything by virtue of checking against the AES SBox
        const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
        let sbox_matrix = _mm256_set1_epi64x(AES_S_BOX_MATRIX);

        for i in 0..NUM_BYTES_WORDS_256 {
            let reference = load_m256i_word(&AES_S_BOX, i);
            let input = load_m256i_word(&inputs, i);
            let result = _mm256_gf2p8affineinv_epi64_epi8(input, sbox_matrix, CONSTANT_BYTE);
            assert_eq_m256i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let matrix = load_m256i_word(&matrices, i);
            let vector = load_m256i_word(&vectors, i);
            let result_zero =
                _mm256_maskz_gf2p8affineinv_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m256i(result_zero, _mm256_setzero_si256());
            let mask_bytes: __mmask32 = 0xFF_0F_F0_00;
            const MASK_WORDS: i32 = 0b11_01_10_00;
            let expected_result = _mm256_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm256_maskz_gf2p8affineinv_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm256_blend_epi32(_mm256_setzero_si256(), expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_256 {
            let left = load_m256i_word(&vectors, i);
            let right = load_m256i_word(&matrices, i);
            let result_left =
                _mm256_mask_gf2p8affineinv_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m256i(result_left, left);
            let mask_bytes: __mmask32 = 0xFF_0F_F0_00;
            const MASK_WORDS: i32 = 0b11_01_10_00;
            let expected_result = _mm256_gf2p8affineinv_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm256_mask_gf2p8affineinv_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm256_blend_epi32(left, expected_result, MASK_WORDS);
            assert_eq_m256i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_gf2p8affineinv_epi64_epi8() {
        let identity: i64 = 0x01_02_04_08_10_20_40_80;
        const IDENTITY_BYTE: i32 = 0;
        const CONSTANT_BYTE: i32 = 0x63;
        let identity = _mm_set1_epi64x(identity);

        // validate inversion
        let (inputs, results) = generate_inv_tests_data();

        for i in 0..NUM_BYTES_WORDS_128 {
            let input = load_m128i_word(&inputs, i);
            let reference = load_m128i_word(&results, i);
            let result = _mm_gf2p8affineinv_epi64_epi8(input, identity, IDENTITY_BYTE);
            let remultiplied = _mm_gf2p8mul_epi8(result, input);
            assert_eq_m128i(remultiplied, reference);
        }

        // validate subsequent affine operation
        let (matrices, vectors, _affine_expected) =
            generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let vector = load_m128i_word(&vectors, i);
            let matrix = load_m128i_word(&matrices, i);

            let inv_vec = _mm_gf2p8affineinv_epi64_epi8(vector, identity, IDENTITY_BYTE);
            let reference = _mm_gf2p8affine_epi64_epi8(inv_vec, matrix, CONSTANT_BYTE);
            let result = _mm_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            assert_eq_m128i(result, reference);
        }

        // validate everything by virtue of checking against the AES SBox
        const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
        let sbox_matrix = _mm_set1_epi64x(AES_S_BOX_MATRIX);

        for i in 0..NUM_BYTES_WORDS_128 {
            let reference = load_m128i_word(&AES_S_BOX, i);
            let input = load_m128i_word(&inputs, i);
            let result = _mm_gf2p8affineinv_epi64_epi8(input, sbox_matrix, CONSTANT_BYTE);
            assert_eq_m128i(result, reference);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let matrix = load_m128i_word(&matrices, i);
            let vector = load_m128i_word(&vectors, i);
            let result_zero = _mm_maskz_gf2p8affineinv_epi64_epi8(0, vector, matrix, CONSTANT_BYTE);
            assert_eq_m128i(result_zero, _mm_setzero_si128());
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8affineinv_epi64_epi8(vector, matrix, CONSTANT_BYTE);
            let result_masked =
                _mm_maskz_gf2p8affineinv_epi64_epi8(mask_bytes, vector, matrix, CONSTANT_BYTE);
            let expected_masked =
                _mm_blend_epi32::<MASK_WORDS>(_mm_setzero_si128(), expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }

    #[simd_test(enable = "avx512gfni,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_gf2p8affineinv_epi64_epi8() {
        const CONSTANT_BYTE: i32 = 0x63;
        let (matrices, vectors, _expected) = generate_affine_mul_test_data(CONSTANT_BYTE as u8);

        for i in 0..NUM_TEST_WORDS_128 {
            let left = load_m128i_word(&vectors, i);
            let right = load_m128i_word(&matrices, i);
            let result_left =
                _mm_mask_gf2p8affineinv_epi64_epi8(left, 0, left, right, CONSTANT_BYTE);
            assert_eq_m128i(result_left, left);
            let mask_bytes: __mmask16 = 0x0F_F0;
            const MASK_WORDS: i32 = 0b01_10;
            let expected_result = _mm_gf2p8affineinv_epi64_epi8(left, right, CONSTANT_BYTE);
            let result_masked =
                _mm_mask_gf2p8affineinv_epi64_epi8(left, mask_bytes, left, right, CONSTANT_BYTE);
            let expected_masked = _mm_blend_epi32::<MASK_WORDS>(left, expected_result);
            assert_eq_m128i(result_masked, expected_masked);
        }
    }
}
