use std::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

use simd_llvm::{simd_cast, simd_shuffle2, simd_shuffle4, simd_shuffle8};
use v128::{f32x4, f64x2, i32x4, i64x2};
use v256::*;

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    a + b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm256_add_ps(a: f32x8, b: f32x8) -> f32x8 {
    a + b
}

/// Compute the bitwise AND of a packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
// Should be 'vandpd' instuction.
// See https://github.com/rust-lang-nursery/stdsimd/issues/71
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_and_pd(a: f64x4, b: f64x4) -> f64x4 {
    let a: u64x4 = mem::transmute(a);
    let b: u64x4 = mem::transmute(b);
    mem::transmute(a & b)
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_and_ps(a: f32x8, b: f32x8) -> f32x8 {
    let a: u32x8 = mem::transmute(a);
    let b: u32x8 = mem::transmute(b);
    mem::transmute(a & b)
}

/// Compute the bitwise OR packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
// Should be 'vorpd' instuction.
// See https://github.com/rust-lang-nursery/stdsimd/issues/71
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm256_or_pd(a: f64x4, b: f64x4) -> f64x4 {
    let a: u64x4 = mem::transmute(a);
    let b: u64x4 = mem::transmute(b);
    mem::transmute(a | b)
}

/// Compute the bitwise OR packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm256_or_ps(a: f32x8, b: f32x8) -> f32x8 {
    let a: u32x8 = mem::transmute(a);
    let b: u32x8 = mem::transmute(b);
    mem::transmute(a | b)
}

/// Shuffle double-precision (64-bit) floating-point elements within 128-bit
/// lanes using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
//#[cfg_attr(test, assert_instr(vshufpd, imm8 = 0x0))] // FIXME
pub unsafe fn _mm256_shuffle_pd(a: f64x4, b: f64x4, imm8: i32) -> f64x4 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, b, [$a, $b, $c, $d]);
        }
    }
    macro_rules! shuffle3 {
        ($a:expr, $b: expr, $c: expr) => {
            match (imm8 >> 3) & 0x1 {
                0 => shuffle4!($a, $b, $c, 6),
                _ => shuffle4!($a, $b, $c, 7),
            }
        }
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 2) & 0x1 {
                0 => shuffle3!($a, $b, 2),
                _ => shuffle3!($a, $b, 3),
            }
        }
    }
    macro_rules! shuffle1 {
        ($a:expr) => {
            match (imm8 >> 1) & 0x1 {
                0 => shuffle2!($a, 4),
                _ => shuffle2!($a, 5),
            }
        }
    }
    match (imm8 >> 0) & 0x1 {
        0 => shuffle1!(0),
        _ => shuffle1!(1),
    }
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating-point elements in `a`
/// and then AND with `b`.
#[inline(always)]
#[target_feature = "+avx"]
// Should be 'vandnpd' instruction.
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm256_andnot_pd(a: f64x4, b: f64x4) -> f64x4 {
    let a: u64x4 = mem::transmute(a);
    let b: u64x4 = mem::transmute(b);
    mem::transmute((!a) & b)
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating-point elements in `a`
/// and then AND with `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm256_andnot_ps(a: f32x8, b: f32x8) -> f32x8 {
    let a: u32x8 = mem::transmute(a);
    let b: u32x8 = mem::transmute(b);
    mem::transmute((!a) & b)
}

/// Compare packed double-precision (64-bit) floating-point elements 
/// in `a` and `b`, and return packed maximum values
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm256_max_pd(a: f64x4, b: f64x4) -> f64x4 {
    maxpd256(a, b)
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and `b`, 
/// and return packed maximum values
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm256_max_ps(a: f32x8, b: f32x8) -> f32x8 {
    maxps256(a, b)
}

/// Compare packed double-precision (64-bit) floating-point elements 
/// in `a` and `b`, and return packed minimum values
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm256_min_pd(a: f64x4, b: f64x4) -> f64x4 {
    minpd256(a, b)
}

/// Compare packed single-precision (32-bit) floating-point elements in `a` and `b`, 
/// and return packed minimum values
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm256_min_ps(a: f32x8, b: f32x8) -> f32x8 {
    minps256(a, b)
}

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmulpd))]
pub unsafe fn _mm256_mul_pd(a: f64x4, b: f64x4) -> f64x4 {
    a * b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmulps))]
pub unsafe fn _mm256_mul_ps(a: f32x8, b: f32x8) -> f32x8 {
    a * b
}

/// Alternatively add and subtract packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddsubpd))]
pub unsafe fn _mm256_addsub_pd(a: f64x4, b: f64x4) -> f64x4 {
    addsubpd256(a, b)
}

/// Alternatively add and subtract packed single-precision (32-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddsubps))]
pub unsafe fn _mm256_addsub_ps(a: f32x8, b: f32x8) -> f32x8 {
    addsubps256(a, b)
}

/// Subtract packed double-precision (64-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsubpd))]
pub unsafe fn _mm256_sub_pd(a: f64x4, b: f64x4) -> f64x4 {
    a - b
}

/// Subtract packed single-precision (32-bit) floating-point elements in `b`
/// from packed elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsubps))]
pub unsafe fn _mm256_sub_ps(a: f32x8, b: f32x8) -> f32x8 {
    a - b
}

/// Compute the division of each of the 8 packed 32-bit floating-point elements
/// in `a` by the corresponding packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vdivps))]
pub unsafe fn _mm256_div_ps(a: f32x8, b: f32x8) -> f32x8 {
    a / b
}

/// Compute the division of each of the 4 packed 64-bit floating-point elements
/// in `a` by the corresponding packed elements in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vdivpd))]
pub unsafe fn _mm256_div_pd(a: f64x4, b: f64x4) -> f64x4 {
    a / b
}


/// Round packed double-precision (64-bit) floating point elements in `a`
/// according to the flag `b`. The value of `b` may be as follows:
///
/// - `0x00`: Round to the nearest whole number.
/// - `0x01`: Round down, toward negative infinity.
/// - `0x02`: Round up, toward positive infinity.
/// - `0x03`: Truncate the values.
///
/// For a complete list of options, check the LLVM docs:
///
/// https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd, b = 0x3))]
pub unsafe fn _mm256_round_pd(a: f64x4, b: i32) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => { roundpd256(a, $imm8) }
    }
    constify_imm8!(b, call)
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd))]
pub unsafe fn _mm256_ceil_pd(a: f64x4) -> f64x4 {
    roundpd256(a, 0x02)
}

/// Round packed double-precision (64-bit) floating point elements in `a` toward
/// negative infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd))]
pub unsafe fn _mm256_floor_pd(a: f64x4) -> f64x4 {
    roundpd256(a, 0x01)
}

/// Round packed single-precision (32-bit) floating point elements in `a`
/// according to the flag `b`. The value of `b` may be as follows:
///
/// - `0x00`: Round to the nearest whole number.
/// - `0x01`: Round down, toward negative infinity.
/// - `0x02`: Round up, toward positive infinity.
/// - `0x03`: Truncate the values.
///
/// For a complete list of options, check the LLVM docs:
///
/// https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundps, b = 0x00))]
pub unsafe fn _mm256_round_ps(a: f32x8, b: i32) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => {
            roundps256(a, $imm8)
        }
    }
    constify_imm8!(b, call)
}

/// Round packed single-precision (32-bit) floating point elements in `a` toward
/// positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundps))]
pub unsafe fn _mm256_ceil_ps(a: f32x8) -> f32x8 {
    roundps256(a, 0x02)
}

/// Round packed single-precision (32-bit) floating point elements in `a` toward
/// negative infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundps))]
pub unsafe fn _mm256_floor_ps(a: f32x8) -> f32x8 {
    roundps256(a, 0x01)
}

/// Return the square root of packed single-precision (32-bit) floating point
/// elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsqrtps))]
pub unsafe fn _mm256_sqrt_ps(a: f32x8) -> f32x8 {
    sqrtps256(a)
}

/// Return the square root of packed double-precision (64-bit) floating point
/// elements in `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vsqrtpd))]
pub unsafe fn _mm256_sqrt_pd(a: f64x4) -> f64x4 {
    sqrtpd256(a)
}

/// Blend packed double-precision (64-bit) floating-point elements from
/// `a` and `b` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vblendpd, imm8 = 9))]
pub unsafe fn _mm256_blend_pd(a: f64x4, b: f64x4, imm8: i32) -> f64x4 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! blend4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, b, [$a, $b, $c, $d]);
        }
    }
    macro_rules! blend3 {
        ($a:expr, $b: expr, $c: expr) => {
            match imm8 & 0x8 {
                0 => blend4!($a, $b, $c, 3),
                _ => blend4!($a, $b, $c, 7),
            }
        }
    }
    macro_rules! blend2 {
        ($a:expr, $b:expr) => {
            match imm8 & 0x4 {
                0 => blend3!($a, $b, 2),
                _ => blend3!($a, $b, 6),
            }
        }
    }
    macro_rules! blend1 {
        ($a:expr) => {
            match imm8 & 0x2 {
                0 => blend2!($a, 1),
                _ => blend2!($a, 5),
            }
        }
    }
    match imm8 & 0x1 {
        0 => blend1!(0),
        _ => blend1!(4),
    }
}

/// Blend packed double-precision (64-bit) floating-point elements from
/// `a` and `b` using `c` as a mask.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vblendvpd))]
pub unsafe fn _mm256_blendv_pd(a: f64x4, b: f64x4, c: f64x4) -> f64x4 {
    vblendvpd(a, b, c)
}

/// Blend packed single-precision (32-bit) floating-point elements from
/// `a` and `b` using `c` as a mask.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vblendvps))]
pub unsafe fn _mm256_blendv_ps(a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
    vblendvps(a, b, c)
}

/// Conditionally multiply the packed single-precision (32-bit) floating-point
/// elements in `a` and `b` using the high 4 bits in `imm8`,
/// sum the four products, and conditionally return the sum
///  using the low 4 bits of `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vdpps, imm8 = 0x0))]
pub unsafe fn _mm256_dp_ps(a: f32x8, b: f32x8, imm8: i32) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => { vdpps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Horizontal addition of adjacent pairs in the two packed vectors
/// of 4 64-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in even locations,
/// while sums of elements from `b` are returned in odd locations.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vhaddpd))]
pub unsafe fn _mm256_hadd_pd(a: f64x4, b: f64x4) -> f64x4 {
    vhaddpd(a, b)
}

/// Horizontal addition of adjacent pairs in the two packed vectors
/// of 8 32-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in locations of
/// indices 0, 1, 4, 5; while sums of elements from `b` are locations
/// 2, 3, 6, 7.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vhaddps))]
pub unsafe fn _mm256_hadd_ps(a: f32x8, b: f32x8) -> f32x8 {
    vhaddps(a, b)
}

/// Horizontal subtraction of adjacent pairs in the two packed vectors
/// of 4 64-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in even locations,
/// while sums of elements from `b` are returned in odd locations.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vhsubpd))]
pub unsafe fn _mm256_hsub_pd(a: f64x4, b: f64x4) -> f64x4 {
    vhsubpd(a, b)
}

/// Horizontal subtraction of adjacent pairs in the two packed vectors
/// of 8 32-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in locations of
/// indices 0, 1, 4, 5; while sums of elements from `b` are locations
/// 2, 3, 6, 7.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vhsubps))]
pub unsafe fn _mm256_hsub_ps(a: f32x8, b: f32x8) -> f32x8 {
    vhsubps(a, b)
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating-point
/// elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
// FIXME Should be 'vxorpd' instruction.
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_xor_pd(a: f64x4, b: f64x4) -> f64x4 {
    let a: u64x4 = mem::transmute(a);
    let b: u64x4 = mem::transmute(b);
    mem::transmute(a ^ b)
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_xor_ps(a: f32x8, b: f32x8) -> f32x8 {
    let a: u32x8 = mem::transmute(a);
    let b: u32x8 = mem::transmute(b);
    mem::transmute(a ^ b)
}

/// Convert packed 32-bit integers in `a` to packed double-precision (64-bit)
/// floating-point elements.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtdq2pd))]
pub unsafe fn _mm256_cvtepi32_pd(a: i32x4) -> f64x4 {
    simd_cast(a)
}

/// Convert packed 32-bit integers in `a` to packed single-precision (32-bit)
/// floating-point elements.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtdq2ps))]
pub unsafe fn _mm256_cvtepi32_ps(a: i32x8) -> f32x8 {
    vcvtdq2ps(a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a`
/// to packed single-precision (32-bit) floating-point elements.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtpd2ps))]
pub unsafe fn _mm256_cvtpd_ps(a: f64x4) -> f32x4 {
    vcvtpd2ps(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a`
/// to packed 32-bit integers.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtps2dq))]
pub unsafe fn _mm256_cvtps_epi32(a: f32x8) -> i32x8 {
    vcvtps2dq(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a`
/// to packed double-precision (64-bit) floating-point elements.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtps2pd))]
pub unsafe fn _mm256_cvtps_pd(a: f32x4) -> f64x4 {
    a.as_f64x4()
}

/// Convert packed double-precision (64-bit) floating-point elements in `a`
/// to packed 32-bit integers with truncation.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvttpd2dq))]
pub unsafe fn _mm256_cvttpd_epi32(a: f64x4) -> i32x4 {
    vcvttpd2dq(a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a`
/// to packed 32-bit integers.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvtpd2dq))]
pub unsafe fn _mm256_cvtpd_epi32(a: f64x4) -> i32x4 {
    vcvtpd2dq(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a`
/// to packed 32-bit integers with truncation.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcvttps2dq))]
pub unsafe fn _mm256_cvttps_epi32(a: f32x8) -> i32x8 {
    vcvttps2dq(a)
}

/// Extract 128 bits (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vextractf128))]
pub unsafe fn _mm256_extractf128_ps(a: f32x8, imm8: i32) -> f32x4 {
    match imm8 & 1 {
        0 => simd_shuffle4(a, _mm256_undefined_ps(), [0, 1, 2, 3]),
        _ => simd_shuffle4(a, _mm256_undefined_ps(), [4, 5, 6, 7]),
    }
}

/// Extract 128 bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vextractf128))]
pub unsafe fn _mm256_extractf128_pd(a: f64x4, imm8: i32) -> f64x2 {
    match imm8 & 1 {
        0 => simd_shuffle2(a, _mm256_undefined_pd(), [0, 1]),
        _ => simd_shuffle2(a, _mm256_undefined_pd(), [2, 3]),
    }
}

/// Extract 128 bits (composed of integer data) from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vextractf128))]
pub unsafe fn _mm256_extractf128_si256(a: i64x4, imm8: i32) -> i64x2 {
    match imm8 & 1 {
        0 => simd_shuffle2(a, _mm256_undefined_si256(), [0, 1]),
        _ => simd_shuffle2(a, _mm256_undefined_si256(), [2, 3]),
    }
}

/// Extract an 8-bit integer from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_extract_epi8(a: i8x32, imm8: i32) -> i32 {
    a.extract(imm8 as u32 & 31) as i32
}

/// Extract a 16-bit integer from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_extract_epi16(a: i16x16, imm8: i32) -> i32 {
    a.extract(imm8 as u32 & 15) as i32
}

/// Extract a 32-bit integer from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_extract_epi32(a: i32x8, imm8: i32) -> i32 {
    a.extract(imm8 as u32 & 7) as i32
}

/// Extract a 64-bit integer from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_extract_epi64(a: i64x4, imm8: i32) -> i32 {
    a.extract(imm8 as u32 & 3) as i32
}

/// Zero the contents of all XMM or YMM registers.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vzeroall))]
pub unsafe fn _mm256_zeroall() {
    vzeroall()
}

/// Zero the upper 128 bits of all YMM registers;
/// the lower 128-bits of the registers are unmodified.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vzeroupper))]
pub unsafe fn _mm256_zeroupper() {
    vzeroupper()
}

/// Shuffle single-precision (32-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilps))]
pub unsafe fn _mm256_permutevar_ps(a: f32x8, b: i32x8) -> f32x8 {
    vpermilps256(a, b)
}

/// Shuffle single-precision (32-bit) floating-point elements in `a`
/// using the control in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilps))]
pub unsafe fn _mm_permutevar_ps(a: f32x4, b: i32x4) -> f32x4 {
    vpermilps(a, b)
}

/// Shuffle single-precision (32-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilps, imm8 = 9))]
pub unsafe fn _mm256_permute_ps(a: f32x8, imm8: i32) -> f32x8 {
    let imm8 = (imm8 & 0xFF) as u8;
    const fn add4(x: u32) -> u32 { x + 4 }
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle8(a, _mm256_undefined_ps(), [
                $a, $b, $c, $d, add4($a), add4($b), add4($c), add4($d)
            ])
        }
    }
    macro_rules! shuffle3 {
        ($a:expr, $b:expr, $c:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle4!($a, $b, $c, 0),
                0b01 => shuffle4!($a, $b, $c, 1),
                0b10 => shuffle4!($a, $b, $c, 2),
                _ => shuffle4!($a, $b, $c, 3),
            }
        }
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle3!($a, $b, 0),
                0b01 => shuffle3!($a, $b, 1),
                0b10 => shuffle3!($a, $b, 2),
                _ => shuffle3!($a, $b, 3),
            }
        }
    }
    macro_rules! shuffle1 {
        ($a:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle2!($a, 0),
                0b01 => shuffle2!($a, 1),
                0b10 => shuffle2!($a, 2),
                _ => shuffle2!($a, 3),
            }
        }
    }
    match (imm8 >> 0) & 0b11 {
        0b00 => shuffle1!(0),
        0b01 => shuffle1!(1),
        0b10 => shuffle1!(2),
        _ => shuffle1!(3),
    }
}

/// Return vector of type `f32x8` with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_ps() -> f32x8 {
    mem::uninitialized()
}

/// Return vector of type `f64x4` with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_pd() -> f64x4 {
    mem::uninitialized()
}

/// Return vector of type `i64x4` with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_si256() -> i64x4 {
    mem::uninitialized()
}

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.addsub.pd.256"]
    fn addsubpd256(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.addsub.ps.256"]
    fn addsubps256(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.max.pd.256"]
    fn maxpd256(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.max.ps.256"]
    fn maxps256(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.min.pd.256"]
    fn minpd256(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.min.ps.256"]
    fn minps256(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.round.pd.256"]
    fn roundpd256(a: f64x4, b: i32) -> f64x4;
    #[link_name = "llvm.x86.avx.round.ps.256"]
    fn roundps256(a: f32x8, b: i32) -> f32x8;
    #[link_name = "llvm.x86.avx.sqrt.pd.256"]
    fn sqrtpd256(a: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.sqrt.ps.256"]
    fn sqrtps256(a: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.blendv.pd.256"]
    fn vblendvpd(a: f64x4, b: f64x4, c: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.blendv.ps.256"]
    fn vblendvps(a: f32x8, b: f32x8, c: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.dp.ps.256"]
    fn vdpps(a: f32x8, b: f32x8, imm8: i32) -> f32x8;
    #[link_name = "llvm.x86.avx.hadd.pd.256"]
    fn vhaddpd(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.hadd.ps.256"]
    fn vhaddps(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.hsub.pd.256"]
    fn vhsubpd(a: f64x4, b: f64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.hsub.ps.256"]
    fn vhsubps(a: f32x8, b: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.cvtdq2.ps.256"]
    fn vcvtdq2ps(a: i32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.cvt.pd2.ps.256"]
    fn vcvtpd2ps(a: f64x4) -> f32x4;
    #[link_name = "llvm.x86.avx.cvt.ps2dq.256"]
    fn vcvtps2dq(a: f32x8) -> i32x8;
    #[link_name = "llvm.x86.avx.cvtt.pd2dq.256"]
    fn vcvttpd2dq(a: f64x4) -> i32x4;
    #[link_name = "llvm.x86.avx.cvt.pd2dq.256"]
    fn vcvtpd2dq(a: f64x4) -> i32x4;
    #[link_name = "llvm.x86.avx.cvtt.ps2dq.256"]
    fn vcvttps2dq(a: f32x8) -> i32x8;
    #[link_name = "llvm.x86.avx.vzeroall"]
    fn vzeroall();
    #[link_name = "llvm.x86.avx.vzeroupper"]
    fn vzeroupper();
    #[link_name = "llvm.x86.avx.vpermilvar.ps.256"]
    fn vpermilps256(a: f32x8, b: i32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.vpermilvar.ps"]
    fn vpermilps(a: f32x4, b: i32x4) -> f32x4;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::{f32x4, f64x2, i32x4, i64x2};
    use v256::*;
    use x86::avx;

    #[simd_test = "avx"]
    unsafe fn _mm256_add_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_add_pd(a, b);
        let e = f64x4::new(6.0, 8.0, 10.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_add_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = avx::_mm256_add_ps(a, b);
        let e = f32x8::new(10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_and_pd() {
        let a = f64x4::splat(1.0);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_and_pd(a, b);
        let e = f64x4::splat(0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_and_ps() {
        let a = f32x8::splat(1.0);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_and_ps(a, b);
        let e = f32x8::splat(0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_or_pd() {
        let a = f64x4::splat(1.0);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_or_pd(a, b);
        let e = f64x4::splat(1.2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_or_ps() {
        let a = f32x8::splat(1.0);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_or_ps(a, b);
        let e = f32x8::splat(1.2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_shuffle_pd() {
        let a = f64x4::new(1.0, 4.0, 5.0, 8.0);
        let b = f64x4::new(2.0, 3.0, 6.0, 7.0);
        let r = avx::_mm256_shuffle_pd(a, b, 0xF);
        let e = f64x4::new(4.0, 3.0, 8.0, 7.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_andnot_pd() {
        let a = f64x4::splat(0.0);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_andnot_pd(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_andnot_ps() {
        let a = f32x8::splat(0.0);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_andnot_ps(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_max_pd() {
        let a = f64x4::new(1.0, 4.0, 5.0, 8.0);
        let b = f64x4::new(2.0, 3.0, 6.0, 7.0);
        let r = avx::_mm256_max_pd(a, b);
        let e = f64x4::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_max_ps() {
        let a = f32x8::new(1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0, 16.0);
        let b = f32x8::new(2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0);
        let r = avx::_mm256_max_ps(a, b);
        let e = f32x8::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_min_pd() {
        let a = f64x4::new(1.0, 4.0, 5.0, 8.0);
        let b = f64x4::new(2.0, 3.0, 6.0, 7.0);
        let r = avx::_mm256_min_pd(a, b);
        let e = f64x4::new(1.0, 3.0, 5.0, 7.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_min_ps() {
        let a = f32x8::new(1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0, 16.0);
        let b = f32x8::new(2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0);
        let r = avx::_mm256_min_ps(a, b);
        let e = f32x8::new(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_mul_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_mul_pd(a, b);
        let e = f64x4::new(5.0, 12.0, 21.0, 32.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_mul_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = f32x8::new(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let r = avx::_mm256_mul_ps(a, b);
        let e = f32x8::new(9.0, 20.0, 33.0, 48.0, 65.0, 84.0, 105.0, 128.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_addsub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_addsub_pd(a, b);
        let e = f64x4::new(-4.0, 8.0, -4.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_addsub_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_addsub_ps(a, b);
        let e = f32x8::new(-4.0, 8.0, -4.0, 12.0, -4.0, 8.0, -4.0, 12.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sub_pd() {
        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_sub_pd(a, b);
        let e = f64x4::new(-4.0,-4.0,-4.0,-4.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sub_ps() {
        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0);
        let r = avx::_mm256_sub_ps(a, b);
        let e = f32x8::new(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_round_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_closest = avx::_mm256_round_pd(a, 0b00000000);
        let result_down = avx::_mm256_round_pd(a, 0b00000001);
        let result_up = avx::_mm256_round_pd(a, 0b00000010);
        let expected_closest = f64x4::new(2.0, 2.0, 4.0, -1.0);
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_floor_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_down = avx::_mm256_floor_pd(a);
        let expected_down = f64x4::new(1.0, 2.0, 3.0, -2.0);
        assert_eq!(result_down, expected_down);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_ceil_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_up = avx::_mm256_ceil_pd(a);
        let expected_up = f64x4::new(2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_round_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_closest = avx::_mm256_round_ps(a, 0b00000000);
        let result_down = avx::_mm256_round_ps(a, 0b00000001);
        let result_up = avx::_mm256_round_ps(a, 0b00000010);
        let expected_closest = f32x8::new(2.0, 2.0, 4.0, -1.0, 2.0, 2.0, 4.0, -1.0);
        let expected_down = f32x8::new(1.0, 2.0, 3.0, -2.0, 1.0, 2.0, 3.0, -2.0);
        let expected_up = f32x8::new(2.0, 3.0, 4.0, -1.0, 2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_floor_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_down = avx::_mm256_floor_ps(a);
        let expected_down = f32x8::new(1.0, 2.0, 3.0, -2.0, 1.0, 2.0, 3.0, -2.0);
        assert_eq!(result_down, expected_down);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_ceil_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_up = avx::_mm256_ceil_ps(a);
        let expected_up = f32x8::new(2.0, 3.0, 4.0, -1.0, 2.0, 3.0, 4.0, -1.0);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sqrt_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_sqrt_pd(a, );
        let e = f64x4::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sqrt_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_sqrt_ps(a);
        let e = f32x8::new(2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_div_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_div_ps(a, b);
        let e = f32x8::new(1.0, 3.0, 8.0, 5.0, 0.5, 1.0, 0.25, 0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_div_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let r = avx::_mm256_div_pd(a, b);
        let e = f64x4::new(1.0, 3.0, 8.0, 5.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blend_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let r = avx::_mm256_blend_pd(a, b, 0x0);
        assert_eq!(r, f64x4::new(4.0, 9.0, 16.0, 25.0));
        let r = avx::_mm256_blend_pd(a, b, 0x3);
        assert_eq!(r, f64x4::new(4.0, 3.0, 16.0, 25.0));
        let r = avx::_mm256_blend_pd(a, b, 0xF);
        assert_eq!(r, f64x4::new(4.0, 3.0, 2.0, 5.0));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blendv_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let c = f64x4::new(0.0, 0.0, !0 as f64, !0 as f64);
        let r = avx::_mm256_blendv_pd(a, b, c);
        let e = f64x4::new(4.0, 9.0, 2.0, 5.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blendv_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let c = f32x8::new(0.0, 0.0, 0.0, 0.0, !0 as f32, !0 as f32, !0 as f32, !0 as f32);
        let r = avx::_mm256_blendv_ps(a, b, c);
        let e = f32x8::new(4.0, 9.0, 16.0, 25.0, 8.0, 9.0, 64.0, 50.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_dp_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_dp_ps(a, b, 0xFF);
        let e = f32x8::new(200.0, 200.0, 200.0, 200.0, 2387.0, 2387.0, 2387.0, 2387.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hadd_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let r = avx::_mm256_hadd_pd(a, b);
        let e = f64x4::new(13.0, 7.0, 41.0, 7.0);
        assert_eq!(r, e);

        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_hadd_pd(a, b);
        let e = f64x4::new(3.0, 11.0, 7.0, 15.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hadd_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_hadd_ps(a, b);
        let e = f32x8::new(13.0, 41.0, 7.0, 7.0, 13.0, 41.0, 17.0, 114.0);
        assert_eq!(r, e);

        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_hadd_ps(a, b);
        let e = f32x8::new(3.0, 7.0, 11.0, 15.0, 3.0, 7.0, 11.0, 15.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hsub_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let r = avx::_mm256_hsub_pd(a, b);
        let e = f64x4::new(-5.0, 1.0, -9.0, -3.0);
        assert_eq!(r, e);

        let a = f64x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f64x4::new(5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_hsub_pd(a, b);
        let e = f64x4::new(-1., -1., -1., -1.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hsub_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_hsub_ps(a, b);
        let e = f32x8::new(-5.0, -9.0, 1.0, -3.0, -5.0, -9.0, -1.0, 14.0);
        assert_eq!(r, e);

        let a = f32x8::new(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0);
        let b = f32x8::new(5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0);
        let r = avx::_mm256_hsub_ps(a, b);
        let e = f32x8::new(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);
        assert_eq!(r, e);
    }


    #[simd_test = "avx"]
    unsafe fn _mm256_xor_pd() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let b = f64x4::splat(0.0);
        let r = avx::_mm256_xor_pd(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_xor_ps() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let b = f32x8::splat(0.0);
        let r = avx::_mm256_xor_ps(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtepi32_pd() {
        let a = i32x4::new(4, 9, 16, 25);
        let r = avx::_mm256_cvtepi32_pd(a);
        let e = f64x4::new(4.0, 9.0, 16.0, 25.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtepi32_ps() {
        let a = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        let r = avx::_mm256_cvtepi32_ps(a);
        let e = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtpd_ps() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvtpd_ps(a);
        let e = f32x4::new(4.0, 9.0, 16.0, 25.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtps_epi32() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvtps_epi32(a);
        let e = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtps_pd() {
        let a = f32x4::new(4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvtps_pd(a);
        let e = f64x4::new(4.0, 9.0, 16.0, 25.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvttpd_epi32() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvttpd_epi32(a);
        let e = i32x4::new(4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtpd_epi32() {
        let a = f64x4::new(4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvtpd_epi32(a);
        let e = i32x4::new(4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvttps_epi32() {
        let a = f32x8::new(4.0, 9.0, 16.0, 25.0, 4.0, 9.0, 16.0, 25.0);
        let r = avx::_mm256_cvttps_epi32(a);
        let e = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_ps() {
        let a = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_extractf128_ps(a, 0);
        let e = f32x4::new(4.0, 3.0, 2.0, 5.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_pd() {
        let a = f64x4::new(4.0, 3.0, 2.0, 5.0);
        let r = avx::_mm256_extractf128_pd(a, 0);
        let e = f64x2::new(4.0, 3.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_si256() {
        let a = i64x4::new(4, 3, 2, 5);
        let r = avx::_mm256_extractf128_si256(a, 0);
        let e = i64x2::new(4, 3);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi8() {
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32);
        let r = avx::_mm256_extract_epi8(a, 0);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx::_mm256_extract_epi16(a, 0);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi32() {
        let a = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx::_mm256_extract_epi32(a, 0);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi64() {
        let a = i64x4::new(0, 1, 2, 3);
        let r = avx::_mm256_extract_epi64(a, 3);
        assert_eq!(r, 3);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_zeroall() {
        avx::_mm256_zeroall();
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_zeroupper() {
        avx::_mm256_zeroupper();
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permutevar_ps() {
        let a = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx::_mm256_permutevar_ps(a, b);
        let e = f32x8::new(3.0, 2.0, 5.0, 4.0, 9.0, 64.0, 50.0, 8.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_permutevar_ps() {
        let a = f32x4::new(4.0, 3.0, 2.0, 5.0);
        let b = i32x4::new(1, 2, 3, 4);
        let r = avx::_mm_permutevar_ps(a, b);
        let e = f32x4::new(3.0, 2.0, 5.0, 4.0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute_ps() {
        let a = f32x8::new(4.0, 3.0, 2.0, 5.0, 8.0, 9.0, 64.0, 50.0);
        let r = avx::_mm256_permute_ps(a, 0x1b);
        let e = f32x8::new(5.0, 2.0, 3.0, 4.0, 50.0, 64.0, 9.0, 8.0);
        assert_eq!(r, e);
    }
}
