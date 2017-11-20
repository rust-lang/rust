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

use core::mem;
use core::ptr;

#[cfg(test)]
use stdsimd_test::assert_instr;

use simd_llvm::{simd_cast, simd_shuffle2, simd_shuffle4, simd_shuffle8};
use v128::{f32x4, f64x2, i32x4, i64x2};
use v256::*;
use x86::{__m128i, __m256i};

/// Add packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    a + b
}

/// Add packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm256_add_ps(a: f32x8, b: f32x8) -> f32x8 {
    a + b
}

/// Compute the bitwise AND of a packed double-precision (64-bit)
/// floating-point elements
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

/// Compute the bitwise AND of packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_and_ps(a: f32x8, b: f32x8) -> f32x8 {
    let a: u32x8 = mem::transmute(a);
    let b: u32x8 = mem::transmute(b);
    mem::transmute(a & b)
}

/// Compute the bitwise OR packed double-precision (64-bit) floating-point
/// elements
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

/// Compute the bitwise OR packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
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
#[cfg_attr(test, assert_instr(vshufpd, imm8 = 0x1))]
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
    match imm8 & 0x1 {
        0 => shuffle1!(0),
        _ => shuffle1!(1),
    }
}

/// Shuffle single-precision (32-bit) floating-point elements in `a` within
/// 128-bit lanes using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vshufps, imm8 = 0x0))]
pub unsafe fn _mm256_shuffle_ps(a: f32x8, b: f32x8, imm8: i32) -> f32x8 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g: expr, $h: expr) => {
            simd_shuffle8(a, b, [$a, $b, $c, $d, $e, $f, $g, $h]);
        }
    }
    macro_rules! shuffle3 {
        ($a:expr, $b: expr, $c: expr, $e:expr, $f:expr, $g:expr) => {
            match (imm8 >> 6) & 0x3 {
                0 => shuffle4!($a, $b, $c, 8, $e, $f, $g, 12),
                1 => shuffle4!($a, $b, $c, 9, $e, $f, $g, 13),
                2 => shuffle4!($a, $b, $c, 10, $e, $f, $g, 14),
                _ => shuffle4!($a, $b, $c, 11, $e, $f, $g, 15),
            }
        }
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr, $e:expr, $f:expr) => {
            match (imm8 >> 4) & 0x3 {
                0 => shuffle3!($a, $b, 8, $e, $f, 12),
                1 => shuffle3!($a, $b, 9, $e, $f, 13),
                2 => shuffle3!($a, $b, 10, $e, $f, 14),
                _ => shuffle3!($a, $b, 11, $e, $f, 15),
            }
        }
    }
    macro_rules! shuffle1 {
        ($a:expr, $e:expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($a, 0, $e, 4),
                1 => shuffle2!($a, 1, $e, 5),
                2 => shuffle2!($a, 2, $e, 6),
                _ => shuffle2!($a, 3, $e, 7),
            }
        }
    }
    match imm8 & 0x3 {
        0 => shuffle1!(0, 4),
        1 => shuffle1!(1, 5),
        2 => shuffle1!(2, 6),
        _ => shuffle1!(3, 7),
    }
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating-point
/// elements in `a`
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

/// Compute the bitwise NOT of packed single-precision (32-bit) floating-point
/// elements in `a`
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

/// Compare packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and return packed maximum values
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

/// Compare packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and return packed minimum values
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

/// Add packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
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
/// For a complete list of options, check [the LLVM docs][llvm_docs].
///
/// [llvm_docs]: https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd, b = 0x3))]
pub unsafe fn _mm256_round_pd(a: f64x4, b: i32) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => { roundpd256(a, $imm8) }
    }
    constify_imm8!(b, call)
}

/// Round packed double-precision (64-bit) floating point elements in `a`
/// toward positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundpd))]
pub unsafe fn _mm256_ceil_pd(a: f64x4) -> f64x4 {
    roundpd256(a, 0x02)
}

/// Round packed double-precision (64-bit) floating point elements in `a`
/// toward negative infinity.
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
/// For a complete list of options, check [the LLVM docs][llvm_docs].
///
/// [llvm_docs]: https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
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

/// Round packed single-precision (32-bit) floating point elements in `a`
/// toward positive infinity.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vroundps))]
pub unsafe fn _mm256_ceil_ps(a: f32x8) -> f32x8 {
    roundps256(a, 0x02)
}

/// Round packed single-precision (32-bit) floating point elements in `a`
/// toward negative infinity.
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

/// Blend packed single-precision (32-bit) floating-point elements from
/// `a` and `b` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vblendps, imm8 = 9))]
pub unsafe fn _mm256_blend_ps(a: f32x8, b: f32x8, imm8: i32) -> f32x8 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! blend4 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr) => {
            simd_shuffle8(a, b, [$a, $b, $c, $d, $e, $f, $g, $h]);
        }
    }
    macro_rules! blend3 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => blend4!($a, $b, $c, $d, $e, $f, 6, 7),
                0b01 => blend4!($a, $b, $c, $d, $e, $f, 14, 7),
                0b10 => blend4!($a, $b, $c, $d, $e, $f, 6, 15),
                _ => blend4!($a, $b, $c, $d, $e, $f, 14, 15),
            }
        }
    }
    macro_rules! blend2 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => blend3!($a, $b, $c, $d, 4, 5),
                0b01 => blend3!($a, $b, $c, $d, 12, 5),
                0b10 => blend3!($a, $b, $c, $d, 4, 13),
                _ => blend3!($a, $b, $c, $d, 12, 13),
            }
        }
    }
    macro_rules! blend1 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => blend2!($a, $b, 2, 3),
                0b01 => blend2!($a, $b, 10, 3),
                0b10 => blend2!($a, $b, 2, 11),
                _ => blend2!($a, $b, 10, 11),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => blend1!(0, 1),
        0b01 => blend1!(8, 1),
        0b10 => blend1!(0, 9),
        _ => blend1!(8, 9),
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

/// Equal (ordered, non-signaling)
pub const _CMP_EQ_OQ: u8 = 0x00;
/// Less-than (ordered, signaling)
pub const _CMP_LT_OS: u8 = 0x01;
/// Less-than-or-equal (ordered, signaling)
pub const _CMP_LE_OS: u8 = 0x02;
/// Unordered (non-signaling)
pub const _CMP_UNORD_Q: u8 = 0x03;
/// Not-equal (unordered, non-signaling)
pub const _CMP_NEQ_UQ: u8 = 0x04;
/// Not-less-than (unordered, signaling)
pub const _CMP_NLT_US: u8 = 0x05;
/// Not-less-than-or-equal (unordered, signaling)
pub const _CMP_NLE_US: u8 = 0x06;
/// Ordered (non-signaling)
pub const _CMP_ORD_Q: u8 = 0x07;
/// Equal (unordered, non-signaling)
pub const _CMP_EQ_UQ: u8 = 0x08;
/// Not-greater-than-or-equal (unordered, signaling)
pub const _CMP_NGE_US: u8 = 0x09;
/// Not-greater-than (unordered, signaling)
pub const _CMP_NGT_US: u8 = 0x0a;
/// False (ordered, non-signaling)
pub const _CMP_FALSE_OQ: u8 = 0x0b;
/// Not-equal (ordered, non-signaling)
pub const _CMP_NEQ_OQ: u8 = 0x0c;
/// Greater-than-or-equal (ordered, signaling)
pub const _CMP_GE_OS: u8 = 0x0d;
/// Greater-than (ordered, signaling)
pub const _CMP_GT_OS: u8 = 0x0e;
/// True (unordered, non-signaling)
pub const _CMP_TRUE_UQ: u8 = 0x0f;
/// Equal (ordered, signaling)
pub const _CMP_EQ_OS: u8 = 0x10;
/// Less-than (ordered, non-signaling)
pub const _CMP_LT_OQ: u8 = 0x11;
/// Less-than-or-equal (ordered, non-signaling)
pub const _CMP_LE_OQ: u8 = 0x12;
/// Unordered (signaling)
pub const _CMP_UNORD_S: u8 = 0x13;
/// Not-equal (unordered, signaling)
pub const _CMP_NEQ_US: u8 = 0x14;
/// Not-less-than (unordered, non-signaling)
pub const _CMP_NLT_UQ: u8 = 0x15;
/// Not-less-than-or-equal (unordered, non-signaling)
pub const _CMP_NLE_UQ: u8 = 0x16;
/// Ordered (signaling)
pub const _CMP_ORD_S: u8 = 0x17;
/// Equal (unordered, signaling)
pub const _CMP_EQ_US: u8 = 0x18;
/// Not-greater-than-or-equal (unordered, non-signaling)
pub const _CMP_NGE_UQ: u8 = 0x19;
/// Not-greater-than (unordered, non-signaling)
pub const _CMP_NGT_UQ: u8 = 0x1a;
/// False (ordered, signaling)
pub const _CMP_FALSE_OS: u8 = 0x1b;
/// Not-equal (ordered, signaling)
pub const _CMP_NEQ_OS: u8 = 0x1c;
/// Greater-than-or-equal (ordered, non-signaling)
pub const _CMP_GE_OQ: u8 = 0x1d;
/// Greater-than (ordered, non-signaling)
pub const _CMP_GT_OQ: u8 = 0x1e;
/// True (unordered, signaling)
pub const _CMP_TRUE_US: u8 = 0x1f;

/// Compare packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
#[cfg_attr(test, assert_instr(vcmpeqpd, imm8 = 0))] // TODO Validate vcmppd
pub unsafe fn _mm_cmp_pd(a: f64x2, b: f64x2, imm8: u8) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => { vcmppd(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
}

/// Compare packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcmpeqpd, imm8 = 0))] // TODO Validate vcmppd
pub unsafe fn _mm256_cmp_pd(a: f64x4, b: f64x4, imm8: u8) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => { vcmppd256(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
}

/// Compare packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx,+sse"]
#[cfg_attr(test, assert_instr(vcmpeqps, imm8 = 0))] // TODO Validate vcmpps
pub unsafe fn _mm_cmp_ps(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => { vcmpps(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
}

/// Compare packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vcmpeqps, imm8 = 0))] // TODO Validate vcmpps
pub unsafe fn _mm256_cmp_ps(a: f32x8, b: f32x8, imm8: u8) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => { vcmpps256(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
}

/// Compare the lower double-precision (64-bit) floating-point element in
/// `a` and `b` based on the comparison operand specified by `imm8`,
/// store the result in the lower element of returned vector,
/// and copy the upper element from `a` to the upper element of returned
/// vector.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
#[cfg_attr(test, assert_instr(vcmpeqsd, imm8 = 0))] // TODO Validate vcmpsd
pub unsafe fn _mm_cmp_sd(a: f64x2, b: f64x2, imm8: u8) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => { vcmpsd(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
}

/// Compare the lower single-precision (32-bit) floating-point element in
/// `a` and `b` based on the comparison operand specified by `imm8`,
/// store the result in the lower element of returned vector,
/// and copy the upper 3 packed elements from `a` to the upper elements of
/// returned vector.
#[inline(always)]
#[target_feature = "+avx,+sse"]
#[cfg_attr(test, assert_instr(vcmpeqss, imm8 = 0))] // TODO Validate vcmpss
pub unsafe fn _mm_cmp_ss(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => { vcmpss(a, b, $imm8) }
    }
    constify_imm6!(imm8, call)
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
pub unsafe fn _mm256_extractf128_si256(a: __m256i, imm8: i32) -> __m128i {
    let b = i64x4::from(_mm256_undefined_si256());
    let dst: i64x2 = match imm8 & 1 {
        0 => simd_shuffle2(i64x4::from(a), b, [0, 1]),
        _ => simd_shuffle2(i64x4::from(a), b, [2, 3]),
    };
    __m128i::from(dst)
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
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle8(a, _mm256_undefined_ps(), [
                $a, $b, $c, $d, $a + 4, $b + 4, $c + 4, $d + 4
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
    match imm8 & 0b11 {
        0b00 => shuffle1!(0),
        0b01 => shuffle1!(1),
        0b10 => shuffle1!(2),
        _ => shuffle1!(3),
    }
}

/// Shuffle single-precision (32-bit) floating-point elements in `a`
/// using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx,+sse"]
#[cfg_attr(test, assert_instr(vpermilps, imm8 = 9))]
pub unsafe fn _mm_permute_ps(a: f32x4, imm8: i32) -> f32x4 {
    use x86::i586::sse::_mm_undefined_ps;

    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, _mm_undefined_ps(), [
                $a, $b, $c, $d
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
    match imm8 & 0b11 {
        0b00 => shuffle1!(0),
        0b01 => shuffle1!(1),
        0b10 => shuffle1!(2),
        _ => shuffle1!(3),
    }
}

/// Shuffle double-precision (64-bit) floating-point elements in `a`
/// within 256-bit lanes using the control in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilpd))]
pub unsafe fn _mm256_permutevar_pd(a: f64x4, b: i64x4) -> f64x4 {
    vpermilpd256(a, b)
}

/// Shuffle double-precision (64-bit) floating-point elements in `a`
/// using the control in `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilpd))]
pub unsafe fn _mm_permutevar_pd(a: f64x2, b: i64x2) -> f64x2 {
    vpermilpd(a, b)
}

/// Shuffle double-precision (64-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpermilpd, imm8 = 0x1))]
pub unsafe fn _mm256_permute_pd(a: f64x4, imm8: i32) -> f64x4 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, _mm256_undefined_pd(), [$a, $b, $c, $d]);
        }
    }
    macro_rules! shuffle3 {
        ($a:expr, $b: expr, $c: expr) => {
            match (imm8 >> 3) & 0x1 {
                0 => shuffle4!($a, $b, $c, 2),
                _ => shuffle4!($a, $b, $c, 3),
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
                0 => shuffle2!($a, 0),
                _ => shuffle2!($a, 1),
            }
        }
    }
    match imm8 & 0x1 {
        0 => shuffle1!(0),
        _ => shuffle1!(1),
    }
}

/// Shuffle double-precision (64-bit) floating-point elements in `a`
/// using the control in `imm8`.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
#[cfg_attr(test, assert_instr(vpermilpd, imm8 = 0x1))]
pub unsafe fn _mm_permute_pd(a: f64x2, imm8: i32) -> f64x2 {
    use x86::i586::sse2::_mm_undefined_pd;

    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle2 {
        ($a:expr, $b:expr) => {
            simd_shuffle2(a, _mm_undefined_pd(), [$a, $b]);
        }
    }
    macro_rules! shuffle1 {
        ($a:expr) => {
            match (imm8 >> 1) & 0x1 {
                0 => shuffle2!($a, 0),
                _ => shuffle2!($a, 1),
            }
        }
    }
    match imm8 & 0x1 {
        0 => shuffle1!(0),
        _ => shuffle1!(1),
    }
}

/// Shuffle 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) selected by `imm8` from `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vperm2f128, imm8 = 0x5))]
pub unsafe fn _mm256_permute2f128_ps(a: f32x8, b: f32x8, imm8: i8) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => { vperm2f128ps256(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Shuffle 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) selected by `imm8` from `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vperm2f128, imm8 = 0x31))]
pub unsafe fn _mm256_permute2f128_pd(a: f64x4, b: f64x4, imm8: i8) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => { vperm2f128pd256(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Shuffle 258-bits (composed of integer data) selected by `imm8`
/// from `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vperm2f128, imm8 = 0x31))]
pub unsafe fn _mm256_permute2f128_si256(
    a: i32x8, b: i32x8, imm8: i8
) -> i32x8 {
    macro_rules! call {
        ($imm8:expr) => { vperm2f128si256(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Broadcast a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm256_broadcast_ss(f: &f32) -> f32x8 {
    f32x8::splat(*f)
}

/// Broadcast a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm_broadcast_ss(f: &f32) -> f32x4 {
    f32x4::splat(*f)
}

/// Broadcast a double-precision (64-bit) floating-point element from memory
/// to all elements of the returned vector.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
pub unsafe fn _mm256_broadcast_sd(f: &f64) -> f64x4 {
    f64x4::splat(*f)
}

/// Broadcast 128 bits from memory (composed of 4 packed single-precision
/// (32-bit) floating-point elements) to all elements of the returned vector.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vbroadcastf128))]
pub unsafe fn _mm256_broadcast_ps(a: &f32x4) -> f32x8 {
    vbroadcastf128ps256(a)
}

/// Broadcast 128 bits from memory (composed of 2 packed double-precision
/// (64-bit) floating-point elements) to all elements of the returned vector.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vbroadcastf128))]
pub unsafe fn _mm256_broadcast_pd(a: &f64x2) -> f64x4 {
    vbroadcastf128pd256(a)
}

/// Copy `a` to result, then insert 128 bits (composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `b` into result
/// at the location specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128, imm8 = 1))]
pub unsafe fn _mm256_insertf128_ps(a: f32x8, b: f32x4, imm8: i32) -> f32x8 {
    let b = _mm256_castps128_ps256(b);
    match imm8 & 1 {
        0 => simd_shuffle8(a, b, [8, 9, 10, 11, 4, 5, 6, 7]),
        _ => simd_shuffle8(a, b, [0, 1, 2, 3, 8, 9, 10, 11]),
    }
}

/// Copy `a` to result, then insert 128 bits (composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `b` into result
/// at the location specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128, imm8 = 1))]
pub unsafe fn _mm256_insertf128_pd(a: f64x4, b: f64x2, imm8: i32) -> f64x4 {
    match imm8 & 1 {
        0 => simd_shuffle4(a, _mm256_castpd128_pd256(b), [4, 5, 2, 3]),
        _ => simd_shuffle4(a, _mm256_castpd128_pd256(b), [0, 1, 4, 5]),
    }
}

/// Copy `a` to result, then insert 128 bits from `b` into result
/// at the location specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128, imm8 = 1))]
pub unsafe fn _mm256_insertf128_si256(
    a: __m256i, b: __m128i, imm8: i32
) -> __m256i {
    let b = i64x4::from(_mm256_castsi128_si256(b));
    let dst: i64x4 = match imm8 & 1 {
        0 => simd_shuffle4(i64x4::from(a), b, [4, 5, 2, 3]),
        _ => simd_shuffle4(i64x4::from(a), b, [0, 1, 4, 5]),
    };
    __m256i::from(dst)
}

/// Copy `a` to result, and insert the 8-bit integer `i` into result
/// at the location specified by `index`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_insert_epi8(a: i8x32, i: i8, index: i32) -> i8x32 {
    let c = a;
    c.replace(index as u32 & 31, i)
}

/// Copy `a` to result, and insert the 16-bit integer `i` into result
/// at the location specified by `index`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_insert_epi16(a: i16x16, i: i16, index: i32) -> i16x16 {
    let c = a;
    c.replace(index as u32 & 15, i)
}

/// Copy `a` to result, and insert the 32-bit integer `i` into result
/// at the location specified by `index`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_insert_epi32(a: i32x8, i: i32, index: i32) -> i32x8 {
    let c = a;
    c.replace(index as u32 & 7, i)
}

/// Copy `a` to result, and insert the 64-bit integer `i` into result
/// at the location specified by `index`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_insert_epi64(a: i64x4, i: i64, index: i32) -> i64x4 {
    let c = a;
    c.replace(index as u32 & 3, i)
}

/// Load 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovapd expected
pub unsafe fn _mm256_load_pd(mem_addr: *const f64) -> f64x4 {
    *(mem_addr as *const f64x4)
}

/// Store 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovapd expected
pub unsafe fn _mm256_store_pd(mem_addr: *const f64, a: f64x4) {
    *(mem_addr as *mut f64x4) = a;
}

/// Load 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))]
pub unsafe fn _mm256_load_ps(mem_addr: *const f32) -> f32x8 {
    *(mem_addr as *const f32x8)
}

/// Store 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))]
pub unsafe fn _mm256_store_ps(mem_addr: *const f32, a: f32x8) {
    *(mem_addr as *mut f32x8) = a;
}

/// Load 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovupd expected
pub unsafe fn _mm256_loadu_pd(mem_addr: *const f64) -> f64x4 {
    let mut dst = f64x4::splat(mem::uninitialized());
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut f64x4 as *mut u8,
        mem::size_of::<f64x4>(),
    );
    dst
}

/// Store 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovupd expected
pub unsafe fn _mm256_storeu_pd(mem_addr: *mut f64, a: f64x4) {
    storeupd256(mem_addr, a);
}

/// Load 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))]
pub unsafe fn _mm256_loadu_ps(mem_addr: *const f32) -> f32x8 {
    let mut dst = f32x8::splat(mem::uninitialized());
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut f32x8 as *mut u8,
        mem::size_of::<f32x8>(),
    );
    dst
}

/// Store 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))]
pub unsafe fn _mm256_storeu_ps(mem_addr: *mut f32, a: f32x8) {
    storeups256(mem_addr, a);
}

/// Load 256-bits of integer data from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovdqa expected
pub unsafe fn _mm256_load_si256(mem_addr: *const __m256i) -> __m256i {
    *mem_addr
}

/// Store 256-bits of integer data from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovdqa expected
pub unsafe fn _mm256_store_si256(mem_addr: *mut __m256i, a: __m256i) {
    *mem_addr = a;
}

/// Load 256-bits of integer data from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu expected
pub unsafe fn _mm256_loadu_si256(mem_addr: *const __m256i) -> __m256i {
    let mut dst = __m256i::splat(mem::uninitialized());
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m256i as *mut u8,
        mem::size_of::<__m256i>(),
    );
    dst
}

/// Store 256-bits of integer data from `a` into memory.
/// 	`mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu expected
pub unsafe fn _mm256_storeu_si256(mem_addr: *mut __m256i, a: __m256i) {
    storeusi256(mem_addr, a);
}

/// Load packed double-precision (64-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
pub unsafe fn _mm256_maskload_pd(mem_addr: *const f64, mask: i64x4) -> f64x4 {
    maskloadpd256(mem_addr as *const i8, mask)
}

/// Store packed double-precision (64-bit) floating-point elements from `a`
/// into memory using `mask`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
pub unsafe fn _mm256_maskstore_pd(mem_addr: *mut f64, mask: i64x4, a: f64x4) {
    maskstorepd256(mem_addr as *mut i8, mask, a);
}

/// Load packed double-precision (64-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
pub unsafe fn _mm_maskload_pd(mem_addr: *const f64, mask: i64x2) -> f64x2 {
    maskloadpd(mem_addr as *const i8, mask)
}

/// Store packed double-precision (64-bit) floating-point elements from `a`
/// into memory using `mask`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
pub unsafe fn _mm_maskstore_pd(mem_addr: *mut f64, mask: i64x2, a: f64x2) {
    maskstorepd(mem_addr as *mut i8, mask, a);
}

/// Load packed single-precision (32-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovps))]
pub unsafe fn _mm256_maskload_ps(mem_addr: *const f32, mask: i32x8) -> f32x8 {
    maskloadps256(mem_addr as *const i8, mask)
}

/// Store packed single-precision (32-bit) floating-point elements from `a`
/// into memory using `mask`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovps))]
pub unsafe fn _mm256_maskstore_ps(mem_addr: *mut f32, mask: i32x8, a: f32x8) {
    maskstoreps256(mem_addr as *mut i8, mask, a);
}

/// Load packed single-precision (32-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovps))]
pub unsafe fn _mm_maskload_ps(mem_addr: *const f32, mask: i32x4) -> f32x4 {
    maskloadps(mem_addr as *const i8, mask)
}

/// Store packed single-precision (32-bit) floating-point elements from `a`
/// into memory using `mask`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmaskmovps))]
pub unsafe fn _mm_maskstore_ps(mem_addr: *mut f32, mask: i32x4, a: f32x4) {
    maskstoreps(mem_addr as *mut i8, mask, a);
}

/// Duplicate odd-indexed single-precision (32-bit) floating-point elements
/// from `a`, and return the results.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovshdup))]
pub unsafe fn _mm256_movehdup_ps(a: f32x8) -> f32x8 {
    simd_shuffle8(a, a, [1, 1, 3, 3, 5, 5, 7, 7])
}

/// Duplicate even-indexed single-precision (32-bit) floating-point elements
/// from `a`, and return the results.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovsldup))]
pub unsafe fn _mm256_moveldup_ps(a: f32x8) -> f32x8 {
    simd_shuffle8(a, a, [0, 0, 2, 2, 4, 4, 6, 6])
}

/// Duplicate even-indexed double-precision (64-bit) floating-point elements
/// from "a", and return the results.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovddup))]
pub unsafe fn _mm256_movedup_pd(a: f64x4) -> f64x4 {
    simd_shuffle4(a, a, [0, 0, 2, 2])
}

/// Load 256-bits of integer data from unaligned memory into result.
/// This intrinsic may perform better than `_mm256_loadu_si256` when the
/// data crosses a cache line boundary.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vlddqu))]
pub unsafe fn _mm256_lddqu_si256(mem_addr: *const i8x32) -> i8x32 {
    vlddqu(mem_addr as *const i8)
}

/// Compute the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`, and return the results. The maximum
/// relative error for this approximation is less than 1.5*2^-12.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vrcpps))]
pub unsafe fn _mm256_rcp_ps(a: f32x8) -> f32x8 {
    vrcpps(a)
}

/// Compute the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`, and return the results.
/// The maximum relative error for this approximation is less than 1.5*2^-12.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vrsqrtps))]
pub unsafe fn _mm256_rsqrt_ps(a: f32x8) -> f32x8 {
    vrsqrtps(a)
}

/// Unpack and interleave double-precision (64-bit) floating-point elements
/// from the high half of each 128-bit lane in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vunpckhpd))]
pub unsafe fn _mm256_unpackhi_pd(a: f64x4, b: f64x4) -> f64x4 {
    simd_shuffle4(a, b, [1, 5, 3, 7])
}

/// Unpack and interleave single-precision (32-bit) floating-point elements
/// from the high half of each 128-bit lane in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vunpckhps))]
pub unsafe fn _mm256_unpackhi_ps(a: f32x8, b: f32x8) -> f32x8 {
    simd_shuffle8(a, b, [2, 10, 3, 11, 6, 14, 7, 15])
}

/// Unpack and interleave double-precision (64-bit) floating-point elements
/// from the low half of each 128-bit lane in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vunpcklpd))]
pub unsafe fn _mm256_unpacklo_pd(a: f64x4, b: f64x4) -> f64x4 {
    simd_shuffle4(a, b, [0, 4, 2, 6])
}

/// Unpack and interleave single-precision (32-bit) floating-point elements
/// from the low half of each 128-bit lane in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vunpcklps))]
pub unsafe fn _mm256_unpacklo_ps(a: f32x8, b: f32x8) -> f32x8 {
    simd_shuffle8(a, b, [0, 8, 1, 9, 4, 12, 5, 13])
}

/// Compute the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Compute the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return the `ZF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vptest))]
pub unsafe fn _mm256_testz_si256(a: i64x4, b: i64x4) -> i32 {
    ptestz256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Compute the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return the `CF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vptest))]
pub unsafe fn _mm256_testc_si256(a: i64x4, b: i64x4) -> i32 {
    ptestc256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Compute the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and
/// `CF` values are zero, otherwise return 0.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vptest))]
pub unsafe fn _mm256_testnzc_si256(a: i64x4, b: i64x4) -> i32 {
    ptestnzc256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm256_testz_pd(a: f64x4, b: f64x4) -> i32 {
    vtestzpd256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm256_testc_pd(a: f64x4, b: f64x4) -> i32 {
    vtestcpd256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm256_testnzc_pd(a: f64x4, b: f64x4) -> i32 {
    vtestnzcpd256(a, b)
}

/// Compute the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm_testz_pd(a: f64x2, b: f64x2) -> i32 {
    vtestzpd(a, b)
}

/// Compute the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm_testc_pd(a: f64x2, b: f64x2) -> i32 {
    vtestcpd(a, b)
}

/// Compute the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestpd))]
pub unsafe fn _mm_testnzc_pd(a: f64x2, b: f64x2) -> i32 {
    vtestnzcpd(a, b)
}

/// Compute the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm256_testz_ps(a: f32x8, b: f32x8) -> i32 {
    vtestzps256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm256_testc_ps(a: f32x8, b: f32x8) -> i32 {
    vtestcps256(a, b)
}

/// Compute the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm256_testnzc_ps(a: f32x8, b: f32x8) -> i32 {
    vtestnzcps256(a, b)
}

/// Compute the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm_testz_ps(a: f32x4, b: f32x4) -> i32 {
    vtestzps(a, b)
}

/// Compute the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm_testc_ps(a: f32x4, b: f32x4) -> i32 {
    vtestcps(a, b)
}

/// Compute the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vtestps))]
pub unsafe fn _mm_testnzc_ps(a: f32x4, b: f32x4) -> i32 {
    vtestnzcps(a, b)
}

/// Set each bit of the returned mask based on the most significant bit of the
/// corresponding packed double-precision (64-bit) floating-point element in
/// `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovmskpd))]
pub unsafe fn _mm256_movemask_pd(a: f64x4) -> i32 {
    movmskpd256(a)
}

/// Set each bit of the returned mask based on the most significant bit of the
/// corresponding packed single-precision (32-bit) floating-point element in
/// `a`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vmovmskps))]
pub unsafe fn _mm256_movemask_ps(a: f32x8) -> i32 {
    movmskps256(a)
}

/// Return vector of type __m256d with all elements set to zero.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vxorps))] // FIXME vxorpd expected
pub unsafe fn _mm256_setzero_pd() -> f64x4 {
    f64x4::new(0., 0., 0., 0.)
}

/// Return vector of type __m256 with all elements set to zero.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_setzero_ps() -> f32x8 {
    f32x8::new(0., 0., 0., 0., 0., 0., 0., 0.)
}

/// Return vector of type __m256i with all elements set to zero.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vxor))]
pub unsafe fn _mm256_setzero_si256() -> __m256i {
    mem::transmute(i64x4::new(0, 0, 0, 0))
}

/// Set packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set_pd(a: f64, b: f64, c: f64, d: f64) -> f64x4 {
    f64x4::new(d, c, b, a)
}

/// Set packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set_ps(
    a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32
) -> f32x8 {
    f32x8::new(h, g, f, e, d, c, b, a)
}

/// Set packed 8-bit integers in returned vector with the supplied values in
/// reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set_epi8(
    e00: i8, e01: i8, e02: i8, e03: i8, e04: i8, e05: i8, e06: i8, e07: i8,
    e08: i8, e09: i8, e10: i8, e11: i8, e12: i8, e13: i8, e14: i8, e15: i8,
    e16: i8, e17: i8, e18: i8, e19: i8, e20: i8, e21: i8, e22: i8, e23: i8,
    e24: i8, e25: i8, e26: i8, e27: i8, e28: i8, e29: i8, e30: i8, e31: i8,
) -> i8x32 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    i8x32::new(
        e31, e30, e29, e28, e27, e26, e25, e24,
        e23, e22, e21, e20, e19, e18, e17, e16,
        e15, e14, e13, e12, e11, e10, e09, e08,
        e07, e06, e05, e04, e03, e02, e01, e00,
    )
}

/// Set packed 16-bit integers in returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set_epi16(
    e00: i16, e01: i16, e02: i16, e03: i16, e04: i16, e05: i16, e06: i16,
    e07: i16, e08: i16, e09: i16, e10: i16, e11: i16, e12: i16, e13: i16,
    e14: i16, e15: i16,
) -> i16x16 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    i16x16::new(
        e15, e14, e13, e12,
        e11, e10, e09, e08,
        e07, e06, e05, e04,
        e03, e02, e01, e00,
    )
}

/// Set packed 32-bit integers in returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set_epi32(
    e0: i32, e1: i32, e2: i32, e3: i32, e4: i32, e5: i32, e6: i32, e7: i32
) -> i32x8 {
    i32x8::new(e7, e6, e5, e4, e3, e2, e1, e0)
}

/// Set packed 64-bit integers in returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set_epi64x(a: i64, b: i64, c: i64, d: i64) -> i64x4 {
    i64x4::new(d, c, b, a)
}

/// Set packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_setr_pd(a: f64, b: f64, c: f64, d: f64) -> f64x4 {
    f64x4::new(a, b, c, d)
}

/// Set packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_setr_ps(
    a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32
) -> f32x8 {
    f32x8::new(a, b, c, d, e, f, g, h)
}

/// Set packed 8-bit integers in returned vector with the supplied values in
/// reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_setr_epi8(
    e00: i8, e01: i8, e02: i8, e03: i8, e04: i8, e05: i8, e06: i8, e07: i8,
    e08: i8, e09: i8, e10: i8, e11: i8, e12: i8, e13: i8, e14: i8, e15: i8,
    e16: i8, e17: i8, e18: i8, e19: i8, e20: i8, e21: i8, e22: i8, e23: i8,
    e24: i8, e25: i8, e26: i8, e27: i8, e28: i8, e29: i8, e30: i8, e31: i8,
) -> i8x32 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    i8x32::new(
        e00, e01, e02, e03, e04, e05, e06, e07,
        e08, e09, e10, e11, e12, e13, e14, e15,
        e16, e17, e18, e19, e20, e21, e22, e23,
        e24, e25, e26, e27, e28, e29, e30, e31,
    )
}

/// Set packed 16-bit integers in returned vector with the supplied values in
/// reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_setr_epi16(
    e00: i16, e01: i16, e02: i16, e03: i16, e04: i16, e05: i16, e06: i16,
    e07: i16, e08: i16, e09: i16, e10: i16, e11: i16, e12: i16, e13: i16,
    e14: i16, e15: i16,
) -> i16x16 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    i16x16::new(
        e00, e01, e02, e03,
        e04, e05, e06, e07,
        e08, e09, e10, e11,
        e12, e13, e14, e15,
    )
}

/// Set packed 32-bit integers in returned vector with the supplied values in
/// reverse order.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_setr_epi32(
    e0: i32, e1: i32, e2: i32, e3: i32, e4: i32, e5: i32, e6: i32, e7: i32
) -> i32x8 {
    i32x8::new(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed 64-bit integers in returned vector with the supplied values in
/// reverse order.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_setr_epi64x(a: i64, b: i64, c: i64, d: i64) -> i64x4 {
    i64x4::new(a, b, c, d)
}

/// Broadcast double-precision (64-bit) floating-point value `a` to all
/// elements of returned vector.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set1_pd(a: f64) -> f64x4 {
    f64x4::new(a, a, a, a)
}

/// Broadcast single-precision (32-bit) floating-point value `a` to all
/// elements of returned vector.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set1_ps(a: f32) -> f32x8 {
    f32x8::new(a, a, a, a, a, a, a, a)
}

/// Broadcast 8-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastb`.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vpshufb))]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set1_epi8(a: i8) -> i8x32 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    i8x32::new(
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
    )
}

/// Broadcast 16-bit integer `a` to all all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastw`.
#[inline(always)]
#[target_feature = "+avx"]
//#[cfg_attr(test, assert_instr(vpshufb))]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set1_epi16(a: i16) -> i16x16 {
    i16x16::new(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a)
}

/// Broadcast 32-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastd`.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_set1_epi32(a: i32) -> i32x8 {
    i32x8::new(a, a, a, a, a, a, a, a)
}

/// Broadcast 64-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastq`.
#[inline(always)]
#[target_feature = "+avx"]
//#[cfg_attr(test, assert_instr(vmovddup))]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set1_epi64x(a: i64) -> i64x4 {
    i64x4::new(a, a, a, a)
}

/// Cast vector of type __m256d to type __m256.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castpd_ps(a: f64x4) -> f32x8 {
    mem::transmute(a)
}

/// Cast vector of type __m256 to type __m256d.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castps_pd(a: f32x8) -> f64x4 {
    mem::transmute(a)
}

/// Casts vector of type __m256 to type __m256i.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castps_si256(a: f32x8) -> __m256i {
    mem::transmute(a)
}

/// Casts vector of type __m256i to type __m256.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castsi256_ps(a: __m256i) -> f32x8 {
    mem::transmute(a)
}

/// Casts vector of type __m256d to type __m256i.
/// This intrinsic is only used for compilation and does not generate any
/// instructions, thus it has zero latency.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castpd_si256(a: f64x4) -> __m256i {
    __m256i::from(a.as_i64x4())
}

/// Casts vector of type __m256i to type __m256d.
/// This intrinsic is only used for compilation and does not generate any
/// instructions, thus it has zero latency.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castsi256_pd(a: __m256i) -> f64x4 {
    simd_cast(i64x4::from(a))
}

/// Casts vector of type __m256 to type __m128.
/// This intrinsic is only used for compilation and does not generate any
/// instructions, thus it has zero latency.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castps256_ps128(a: f32x8) -> f32x4 {
    simd_shuffle4(a, a, [0, 1, 2, 3])
}

/// Casts vector of type __m256d to type __m128d.
/// This intrinsic is only used for compilation and does not generate any
/// instructions, thus it has zero latency.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castpd256_pd128(a: f64x4) -> f64x2 {
    simd_shuffle2(a, a, [0, 1])
}

/// Casts vector of type __m256i to type __m128i.
/// This intrinsic is only used for compilation and does not generate any
/// instructions, thus it has zero latency.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castsi256_si128(a: __m256i) -> __m128i {
    let a = i64x4::from(a);
    let dst: i64x2 = simd_shuffle2(a, a, [0, 1]);
    __m128i::from(dst)
}

/// Casts vector of type __m128 to type __m256;
/// the upper 128 bits of the result are undefined.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castps128_ps256(a: f32x4) -> f32x8 {
    // FIXME simd_shuffle8(a, a, [0, 1, 2, 3, -1, -1, -1, -1])
    simd_shuffle8(a, a, [0, 1, 2, 3, 0, 0, 0, 0])
}

/// Casts vector of type __m128d to type __m256d;
/// the upper 128 bits of the result are undefined.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castpd128_pd256(a: f64x2) -> f64x4 {
    // FIXME simd_shuffle4(a, a, [0, 1, -1, -1])
    simd_shuffle4(a, a, [0, 1, 0, 0])
}

/// Casts vector of type __m128i to type __m256i;
/// the upper 128 bits of the result are undefined.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_castsi128_si256(a: __m128i) -> __m256i {
    let a = i64x2::from(a);
    // FIXME simd_shuffle4(a, a, [0, 1, -1, -1])
    let dst: i64x4 = simd_shuffle4(a, a, [0, 1, 0, 0]);
    __m256i::from(dst)
}

/// Constructs a 256-bit floating-point vector of [8 x float] from a
/// 128-bit floating-point vector of [4 x float]. The lower 128 bits contain
/// the value of the source vector. The upper 128 bits are set to zero.
#[inline(always)]
#[target_feature = "+avx,+sse"]
pub unsafe fn _mm256_zextps128_ps256(a: f32x4) -> f32x8 {
    use x86::i586::sse::_mm_setzero_ps;
    simd_shuffle8(a, _mm_setzero_ps(), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Constructs a 256-bit integer vector from a 128-bit integer vector.
/// The lower 128 bits contain the value of the source vector. The upper
/// 128 bits are set to zero.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_zextsi128_si256(a: __m128i) -> __m256i {
    use x86::i586::sse2::_mm_setzero_si128;
    let b = mem::transmute(_mm_setzero_si128());
    let dst: i64x4 = simd_shuffle4(i64x2::from(a), b, [0, 1, 2, 3]);
    __m256i::from(dst)
}

/// Constructs a 256-bit floating-point vector of [4 x double] from a
/// 128-bit floating-point vector of [2 x double]. The lower 128 bits
/// contain the value of the source vector. The upper 128 bits are set
/// to zero.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_zextpd128_pd256(a: f64x2) -> f64x4 {
    use x86::i586::sse2::_mm_setzero_pd;
    simd_shuffle4(a, _mm_setzero_pd(), [0, 1, 2, 3])
}

/// Return vector of type `f32x8` with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_ps() -> f32x8 {
    f32x8::splat(mem::uninitialized())
}

/// Return vector of type `f64x4` with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_pd() -> f64x4 {
    f64x4::splat(mem::uninitialized())
}

/// Return vector of type __m256i with undefined elements.
#[inline(always)]
#[target_feature = "+avx"]
pub unsafe fn _mm256_undefined_si256() -> __m256i {
    mem::transmute(i64x4::splat(mem::uninitialized()))
}

/// Set packed __m256 returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set_m128(hi: f32x4, lo: f32x4) -> f32x8 {
    simd_shuffle8(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Set packed __m256d returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set_m128d(hi: f64x2, lo: f64x2) -> f64x4 {
    let hi: f32x4 = mem::transmute(hi);
    let lo: f32x4 = mem::transmute(lo);
    mem::transmute(_mm256_set_m128(hi, lo))
}

/// Set packed __m256i returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    let hi: f32x4 = mem::transmute(hi);
    let lo: f32x4 = mem::transmute(lo);
    mem::transmute(_mm256_set_m128(hi, lo))
}

/// Set packed __m256 returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_setr_m128(lo: f32x4, hi: f32x4) -> f32x8 {
    _mm256_set_m128(hi, lo)
}

/// Set packed __m256d returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_setr_m128d(lo: f64x2, hi: f64x2) -> f64x4 {
    _mm256_set_m128d(hi, lo)
}

/// Set packed __m256i returned vector with the supplied values.
#[inline(always)]
#[target_feature = "+avx"]
#[cfg_attr(test, assert_instr(vinsertf128))]
pub unsafe fn _mm256_setr_m128i(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_set_m128i(hi, lo)
}

/// Load two 128-bit values (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse"]
pub unsafe fn _mm256_loadu2_m128(
    hiaddr: *const f32, loaddr: *const f32
) -> f32x8 {
    use x86::i586::sse::_mm_loadu_ps;
    let a = _mm256_castps128_ps256(_mm_loadu_ps(loaddr));
    _mm256_insertf128_ps(a, _mm_loadu_ps(hiaddr), 1)
}

/// Load two 128-bit values (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_loadu2_m128d(
    hiaddr: *const f64, loaddr: *const f64
) -> f64x4 {
    use x86::i586::sse2::_mm_loadu_pd;
    let a = _mm256_castpd128_pd256(_mm_loadu_pd(loaddr));
    _mm256_insertf128_pd(a, _mm_loadu_pd(hiaddr), 1)
}

/// Load two 128-bit values (composed of integer data) from memory, and combine
/// them into a 256-bit value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_loadu2_m128i(
    hiaddr: *const __m128i, loaddr: *const __m128i
) -> __m256i {
    use x86::i586::sse2::_mm_loadu_si128;
    let a = _mm256_castsi128_si256(_mm_loadu_si128(loaddr));
    _mm256_insertf128_si256(a, _mm_loadu_si128(hiaddr), 1)
}

/// Store the high and low 128-bit halves (each composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse"]
pub unsafe fn _mm256_storeu2_m128(
    hiaddr: *mut f32, loaddr: *mut f32, a: f32x8
) {
    use x86::i586::sse::_mm_storeu_ps;
    let lo = _mm256_castps256_ps128(a);
    _mm_storeu_ps(loaddr, lo);
    let hi = _mm256_extractf128_ps(a, 1);
    _mm_storeu_ps(hiaddr, hi);
}

/// Store the high and low 128-bit halves (each composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_storeu2_m128d(
    hiaddr: *mut f64, loaddr: *mut f64, a: f64x4
) {
    use x86::i586::sse2::_mm_storeu_pd;
    let lo = _mm256_castpd256_pd128(a);
    _mm_storeu_pd(loaddr, lo);
    let hi = _mm256_extractf128_pd(a, 1);
    _mm_storeu_pd(hiaddr, hi);
}

/// Store the high and low 128-bit halves (each composed of integer data) from
/// `a` into memory two different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
#[inline(always)]
#[target_feature = "+avx,+sse2"]
pub unsafe fn _mm256_storeu2_m128i(
    hiaddr: *mut __m128i, loaddr: *mut __m128i, a: __m256i
) {
    use x86::i586::sse2::_mm_storeu_si128;
    let lo = _mm256_castsi256_si128(a);
    _mm_storeu_si128(loaddr, lo);
    let hi = _mm256_extractf128_si256(a, 1);
    _mm_storeu_si128(hiaddr, hi);
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
    #[link_name = "llvm.x86.sse2.cmp.pd"]
    fn vcmppd(a: f64x2, b: f64x2, imm8: u8) -> f64x2;
    #[link_name = "llvm.x86.avx.cmp.pd.256"]
    fn vcmppd256(a: f64x4, b: f64x4, imm8: u8) -> f64x4;
    #[link_name = "llvm.x86.sse.cmp.ps"]
    fn vcmpps(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
    #[link_name = "llvm.x86.avx.cmp.ps.256"]
    fn vcmpps256(a: f32x8, b: f32x8, imm8: u8) -> f32x8;
    #[link_name = "llvm.x86.sse2.cmp.sd"]
    fn vcmpsd(a: f64x2, b: f64x2, imm8: u8) -> f64x2;
    #[link_name = "llvm.x86.sse.cmp.ss"]
    fn vcmpss(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
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
    #[link_name = "llvm.x86.avx.vpermilvar.pd.256"]
    fn vpermilpd256(a: f64x4, b: i64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.vpermilvar.pd"]
    fn vpermilpd(a: f64x2, b: i64x2) -> f64x2;
    #[link_name = "llvm.x86.avx.vperm2f128.ps.256"]
    fn vperm2f128ps256(a: f32x8, b: f32x8, imm8: i8) -> f32x8;
    #[link_name = "llvm.x86.avx.vperm2f128.pd.256"]
    fn vperm2f128pd256(a: f64x4, b: f64x4, imm8: i8) -> f64x4;
    #[link_name = "llvm.x86.avx.vperm2f128.si.256"]
    fn vperm2f128si256(a: i32x8, b: i32x8, imm8: i8) -> i32x8;
    #[link_name = "llvm.x86.avx.vbroadcastf128.ps.256"]
    fn vbroadcastf128ps256(a: &f32x4) -> f32x8;
    #[link_name = "llvm.x86.avx.vbroadcastf128.pd.256"]
    fn vbroadcastf128pd256(a: &f64x2) -> f64x4;
    #[link_name = "llvm.x86.avx.storeu.pd.256"]
    fn storeupd256(mem_addr: *mut f64, a: f64x4);
    #[link_name = "llvm.x86.avx.storeu.ps.256"]
    fn storeups256(mem_addr: *mut f32, a: f32x8);
    #[link_name = "llvm.x86.avx.storeu.si.256"]
    fn storeusi256(mem_addr: *mut __m256i, a: __m256i);
    #[link_name = "llvm.x86.avx.maskload.pd.256"]
    fn maskloadpd256(mem_addr: *const i8, mask: i64x4) -> f64x4;
    #[link_name = "llvm.x86.avx.maskstore.pd.256"]
    fn maskstorepd256(mem_addr: *mut i8, mask: i64x4, a: f64x4);
    #[link_name = "llvm.x86.avx.maskload.pd"]
    fn maskloadpd(mem_addr: *const i8, mask: i64x2) -> f64x2;
    #[link_name = "llvm.x86.avx.maskstore.pd"]
    fn maskstorepd(mem_addr: *mut i8, mask: i64x2, a: f64x2);
    #[link_name = "llvm.x86.avx.maskload.ps.256"]
    fn maskloadps256(mem_addr: *const i8, mask: i32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.maskstore.ps.256"]
    fn maskstoreps256(mem_addr: *mut i8, mask: i32x8, a: f32x8);
    #[link_name = "llvm.x86.avx.maskload.ps"]
    fn maskloadps(mem_addr: *const i8, mask: i32x4) -> f32x4;
    #[link_name = "llvm.x86.avx.maskstore.ps"]
    fn maskstoreps(mem_addr: *mut i8, mask: i32x4, a: f32x4);
    #[link_name = "llvm.x86.avx.ldu.dq.256"]
    fn vlddqu(mem_addr: *const i8) -> i8x32;
    #[link_name = "llvm.x86.avx.rcp.ps.256"]
    fn vrcpps(a: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.rsqrt.ps.256"]
    fn vrsqrtps(a: f32x8) -> f32x8;
    #[link_name = "llvm.x86.avx.ptestz.256"]
    fn ptestz256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.ptestc.256"]
    fn ptestc256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.ptestnzc.256"]
    fn ptestnzc256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.pd.256"]
    fn vtestzpd256(a: f64x4, b: f64x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.pd.256"]
    fn vtestcpd256(a: f64x4, b: f64x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.pd.256"]
    fn vtestnzcpd256(a: f64x4, b: f64x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.pd"]
    fn vtestzpd(a: f64x2, b: f64x2) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.pd"]
    fn vtestcpd(a: f64x2, b: f64x2) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.pd"]
    fn vtestnzcpd(a: f64x2, b: f64x2) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.ps.256"]
    fn vtestzps256(a: f32x8, b: f32x8) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.ps.256"]
    fn vtestcps256(a: f32x8, b: f32x8) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.ps.256"]
    fn vtestnzcps256(a: f32x8, b: f32x8) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.ps"]
    fn vtestzps(a: f32x4, b: f32x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.ps"]
    fn vtestcps(a: f32x4, b: f32x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.ps"]
    fn vtestnzcps(a: f32x4, b: f32x4) -> i32;
    #[link_name = "llvm.x86.avx.movmsk.pd.256"]
    fn movmskpd256(a: f64x4) -> i32;
    #[link_name = "llvm.x86.avx.movmsk.ps.256"]
    fn movmskps256(a: f32x8) -> i32;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;
    use test::black_box; // Used to inhibit constant-folding.

    use v128::{f32x4, f64x2, i32x4, i64x2, i8x16};
    use v256::*;
    use x86::i586::avx;
    use x86::{__m128i, __m256i};

    #[simd_test = "avx"]
    unsafe fn _mm256_add_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_add_pd(a, b);
        let e = f64x4::new(6., 8., 10., 12.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_add_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = f32x8::new(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = avx::_mm256_add_ps(a, b);
        let e = f32x8::new(10., 12., 14., 16., 18., 20., 22., 24.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_and_pd() {
        let a = f64x4::splat(1.);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_and_pd(a, b);
        let e = f64x4::splat(0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_and_ps() {
        let a = f32x8::splat(1.);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_and_ps(a, b);
        let e = f32x8::splat(0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_or_pd() {
        let a = f64x4::splat(1.);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_or_pd(a, b);
        let e = f64x4::splat(1.2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_or_ps() {
        let a = f32x8::splat(1.);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_or_ps(a, b);
        let e = f32x8::splat(1.2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_shuffle_pd() {
        let a = f64x4::new(1., 4., 5., 8.);
        let b = f64x4::new(2., 3., 6., 7.);
        let r = avx::_mm256_shuffle_pd(a, b, 0xF);
        let e = f64x4::new(4., 3., 8., 7.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_shuffle_ps() {
        let a = f32x8::new(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = f32x8::new(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = avx::_mm256_shuffle_ps(a, b, 0x0F);
        let e = f32x8::new(8., 8., 2., 2., 16., 16., 10., 10.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_andnot_pd() {
        let a = f64x4::splat(0.);
        let b = f64x4::splat(0.6);
        let r = avx::_mm256_andnot_pd(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_andnot_ps() {
        let a = f32x8::splat(0.);
        let b = f32x8::splat(0.6);
        let r = avx::_mm256_andnot_ps(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_max_pd() {
        let a = f64x4::new(1., 4., 5., 8.);
        let b = f64x4::new(2., 3., 6., 7.);
        let r = avx::_mm256_max_pd(a, b);
        let e = f64x4::new(2., 4., 6., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_max_ps() {
        let a = f32x8::new(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = f32x8::new(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = avx::_mm256_max_ps(a, b);
        let e = f32x8::new(2., 4., 6., 8., 10., 12., 14., 16.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_min_pd() {
        let a = f64x4::new(1., 4., 5., 8.);
        let b = f64x4::new(2., 3., 6., 7.);
        let r = avx::_mm256_min_pd(a, b);
        let e = f64x4::new(1., 3., 5., 7.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_min_ps() {
        let a = f32x8::new(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = f32x8::new(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = avx::_mm256_min_ps(a, b);
        let e = f32x8::new(1., 3., 5., 7., 9., 11., 13., 15.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_mul_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_mul_pd(a, b);
        let e = f64x4::new(5., 12., 21., 32.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_mul_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = f32x8::new(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = avx::_mm256_mul_ps(a, b);
        let e = f32x8::new(9., 20., 33., 48., 65., 84., 105., 128.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_addsub_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_addsub_pd(a, b);
        let e = f64x4::new(-4., 8., -4., 12.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_addsub_ps() {
        let a = f32x8::new(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = f32x8::new(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = avx::_mm256_addsub_ps(a, b);
        let e = f32x8::new(-4., 8., -4., 12., -4., 8., -4., 12.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sub_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_sub_pd(a, b);
        let e = f64x4::new(-4., -4., -4., -4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sub_ps() {
        let a = f32x8::new(1., 2., 3., 4., -1., -2., -3., -4.);
        let b = f32x8::new(5., 6., 7., 8., 3., 2., 1., 0.);
        let r = avx::_mm256_sub_ps(a, b);
        let e = f32x8::new(-4., -4., -4., -4., -4., -4., -4., -4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_round_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_closest = avx::_mm256_round_pd(a, 0b00000000);
        let result_down = avx::_mm256_round_pd(a, 0b00000001);
        let result_up = avx::_mm256_round_pd(a, 0b00000010);
        let expected_closest = f64x4::new(2., 2., 4., -1.);
        let expected_down = f64x4::new(1., 2., 3., -2.);
        let expected_up = f64x4::new(2., 3., 4., -1.);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_floor_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_down = avx::_mm256_floor_pd(a);
        let expected_down = f64x4::new(1., 2., 3., -2.);
        assert_eq!(result_down, expected_down);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_ceil_pd() {
        let a = f64x4::new(1.55, 2.2, 3.99, -1.2);
        let result_up = avx::_mm256_ceil_pd(a);
        let expected_up = f64x4::new(2., 3., 4., -1.);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_round_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_closest = avx::_mm256_round_ps(a, 0b00000000);
        let result_down = avx::_mm256_round_ps(a, 0b00000001);
        let result_up = avx::_mm256_round_ps(a, 0b00000010);
        let expected_closest = f32x8::new(2., 2., 4., -1., 2., 2., 4., -1.);
        let expected_down = f32x8::new(1., 2., 3., -2., 1., 2., 3., -2.);
        let expected_up = f32x8::new(2., 3., 4., -1., 2., 3., 4., -1.);
        assert_eq!(result_closest, expected_closest);
        assert_eq!(result_down, expected_down);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_floor_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_down = avx::_mm256_floor_ps(a);
        let expected_down = f32x8::new(1., 2., 3., -2., 1., 2., 3., -2.);
        assert_eq!(result_down, expected_down);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_ceil_ps() {
        let a = f32x8::new(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_up = avx::_mm256_ceil_ps(a);
        let expected_up = f32x8::new(2., 3., 4., -1., 2., 3., 4., -1.);
        assert_eq!(result_up, expected_up);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sqrt_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_sqrt_pd(a);
        let e = f64x4::new(2., 3., 4., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_sqrt_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = avx::_mm256_sqrt_ps(a);
        let e = f32x8::new(2., 3., 4., 5., 2., 3., 4., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_div_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_div_ps(a, b);
        let e = f32x8::new(1., 3., 8., 5., 0.5, 1., 0.25, 0.5);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_div_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_div_pd(a, b);
        let e = f64x4::new(1., 3., 8., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blend_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_blend_pd(a, b, 0x0);
        assert_eq!(r, f64x4::new(4., 9., 16., 25.));
        let r = avx::_mm256_blend_pd(a, b, 0x3);
        assert_eq!(r, f64x4::new(4., 3., 16., 25.));
        let r = avx::_mm256_blend_pd(a, b, 0xF);
        assert_eq!(r, f64x4::new(4., 3., 2., 5.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blend_ps() {
        let a = f32x8::new(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = f32x8::new(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = avx::_mm256_blend_ps(a, b, 0x0);
        assert_eq!(r, f32x8::new(1., 4., 5., 8., 9., 12., 13., 16.));
        let r = avx::_mm256_blend_ps(a, b, 0x3);
        assert_eq!(r, f32x8::new(2., 3., 5., 8., 9., 12., 13., 16.));
        let r = avx::_mm256_blend_ps(a, b, 0xF);
        assert_eq!(r, f32x8::new(2., 3., 6., 7., 9., 12., 13., 16.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blendv_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::new(4., 3., 2., 5.);
        let c = f64x4::new(0., 0., !0 as f64, !0 as f64);
        let r = avx::_mm256_blendv_pd(a, b, c);
        let e = f64x4::new(4., 9., 2., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_blendv_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let c = f32x8::new(
            0., 0., 0., 0., !0 as f32, !0 as f32, !0 as f32, !0 as f32,
        );
        let r = avx::_mm256_blendv_ps(a, b, c);
        let e = f32x8::new(4., 9., 16., 25., 8., 9., 64., 50.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_dp_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_dp_ps(a, b, 0xFF);
        let e = f32x8::new(200., 200., 200., 200., 2387., 2387., 2387., 2387.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hadd_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_hadd_pd(a, b);
        let e = f64x4::new(13., 7., 41., 7.);
        assert_eq!(r, e);

        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_hadd_pd(a, b);
        let e = f64x4::new(3., 11., 7., 15.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hadd_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_hadd_ps(a, b);
        let e = f32x8::new(13., 41., 7., 7., 13., 41., 17., 114.);
        assert_eq!(r, e);

        let a = f32x8::new(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = f32x8::new(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = avx::_mm256_hadd_ps(a, b);
        let e = f32x8::new(3., 7., 11., 15., 3., 7., 11., 15.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hsub_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_hsub_pd(a, b);
        let e = f64x4::new(-5., 1., -9., -3.);
        assert_eq!(r, e);

        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_hsub_pd(a, b);
        let e = f64x4::new(-1., -1., -1., -1.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_hsub_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_hsub_ps(a, b);
        let e = f32x8::new(-5., -9., 1., -3., -5., -9., -1., 14.);
        assert_eq!(r, e);

        let a = f32x8::new(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = f32x8::new(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = avx::_mm256_hsub_ps(a, b);
        let e = f32x8::new(-1., -1., -1., -1., -1., -1., -1., -1.);
        assert_eq!(r, e);
    }


    #[simd_test = "avx"]
    unsafe fn _mm256_xor_pd() {
        let a = f64x4::new(4., 9., 16., 25.);
        let b = f64x4::splat(0.);
        let r = avx::_mm256_xor_pd(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_xor_ps() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = f32x8::splat(0.);
        let r = avx::_mm256_xor_ps(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_cmp_pd() {
        let a = f64x2::new(4., 9.);
        let b = f64x2::new(4., 3.);
        let r = avx::_mm_cmp_pd(a, b, avx::_CMP_GE_OS);
        assert!(r.extract(0).is_nan());
        assert!(r.extract(1).is_nan());
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cmp_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_cmp_pd(a, b, avx::_CMP_GE_OS);
        let e = f64x4::splat(0.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_cmp_ps() {
        let a = f32x4::new(4., 3., 2., 5.);
        let b = f32x4::new(4., 9., 16., 25.);
        let r = avx::_mm_cmp_ps(a, b, avx::_CMP_GE_OS);
        assert!(r.extract(0).is_nan());
        assert_eq!(r.extract(1), 0.);
        assert_eq!(r.extract(2), 0.);
        assert_eq!(r.extract(3), 0.);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cmp_ps() {
        let a = f32x8::new(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = f32x8::new(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = avx::_mm256_cmp_ps(a, b, avx::_CMP_GE_OS);
        let e = f32x8::splat(0.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_cmp_sd() {
        let a = f64x2::new(4., 9.);
        let b = f64x2::new(4., 3.);
        let r = avx::_mm_cmp_sd(a, b, avx::_CMP_GE_OS);
        assert!(r.extract(0).is_nan());
        assert_eq!(r.extract(1), 9.);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_cmp_ss() {
        let a = f32x4::new(4., 3., 2., 5.);
        let b = f32x4::new(4., 9., 16., 25.);
        let r = avx::_mm_cmp_ss(a, b, avx::_CMP_GE_OS);
        assert!(r.extract(0).is_nan());
        assert_eq!(r.extract(1), 3.);
        assert_eq!(r.extract(2), 2.);
        assert_eq!(r.extract(3), 5.);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtepi32_pd() {
        let a = i32x4::new(4, 9, 16, 25);
        let r = avx::_mm256_cvtepi32_pd(a);
        let e = f64x4::new(4., 9., 16., 25.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtepi32_ps() {
        let a = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        let r = avx::_mm256_cvtepi32_ps(a);
        let e = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtpd_ps() {
        let a = f64x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_cvtpd_ps(a);
        let e = f32x4::new(4., 9., 16., 25.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtps_epi32() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = avx::_mm256_cvtps_epi32(a);
        let e = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtps_pd() {
        let a = f32x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_cvtps_pd(a);
        let e = f64x4::new(4., 9., 16., 25.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvttpd_epi32() {
        let a = f64x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_cvttpd_epi32(a);
        let e = i32x4::new(4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvtpd_epi32() {
        let a = f64x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_cvtpd_epi32(a);
        let e = i32x4::new(4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_cvttps_epi32() {
        let a = f32x8::new(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = avx::_mm256_cvttps_epi32(a);
        let e = i32x8::new(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_ps() {
        let a = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_extractf128_ps(a, 0);
        let e = f32x4::new(4., 3., 2., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_pd() {
        let a = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_extractf128_pd(a, 0);
        let e = f64x2::new(4., 3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extractf128_si256() {
        let a = i64x4::new(4, 3, 2, 5);
        let r = avx::_mm256_extractf128_si256(__m256i::from(a), 0);
        let e = i64x2::new(4, 3);
        assert_eq!(r, __m128i::from(e));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = avx::_mm256_extract_epi8(a, 0);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_extract_epi16() {
        let a =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
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
        let a = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx::_mm256_permutevar_ps(a, b);
        let e = f32x8::new(3., 2., 5., 4., 9., 64., 50., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_permutevar_ps() {
        let a = f32x4::new(4., 3., 2., 5.);
        let b = i32x4::new(1, 2, 3, 4);
        let r = avx::_mm_permutevar_ps(a, b);
        let e = f32x4::new(3., 2., 5., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute_ps() {
        let a = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = avx::_mm256_permute_ps(a, 0x1b);
        let e = f32x8::new(5., 2., 3., 4., 50., 64., 9., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_permute_ps() {
        let a = f32x4::new(4., 3., 2., 5.);
        let r = avx::_mm_permute_ps(a, 0x1b);
        let e = f32x4::new(5., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permutevar_pd() {
        let a = f64x4::new(4., 3., 2., 5.);
        let b = i64x4::new(1, 2, 3, 4);
        let r = avx::_mm256_permutevar_pd(a, b);
        let e = f64x4::new(4., 3., 5., 2.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_permutevar_pd() {
        let a = f64x2::new(4., 3.);
        let b = i64x2::new(3, 0);
        let r = avx::_mm_permutevar_pd(a, b);
        let e = f64x2::new(3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute_pd() {
        let a = f64x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_permute_pd(a, 5);
        let e = f64x4::new(3., 4., 5., 2.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_permute_pd() {
        let a = f64x2::new(4., 3.);
        let r = avx::_mm_permute_pd(a, 1);
        let e = f64x2::new(3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute2f128_ps() {
        let a = f32x8::new(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = f32x8::new(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = avx::_mm256_permute2f128_ps(a, b, 0x13);
        let e = f32x8::new(5., 6., 7., 8., 1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute2f128_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_permute2f128_pd(a, b, 0x31);
        let e = f64x4::new(3., 4., 7., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_permute2f128_si256() {
        let a = i32x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        let b = i32x8::new(5, 6, 7, 8, 5, 6, 7, 8);
        let r = avx::_mm256_permute2f128_si256(a, b, 0x20);
        let e = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_broadcast_ss() {
        let r = avx::_mm256_broadcast_ss(&3.);
        let e = f32x8::splat(3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_broadcast_ss() {
        let r = avx::_mm_broadcast_ss(&3.);
        let e = f32x4::splat(3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_broadcast_sd() {
        let r = avx::_mm256_broadcast_sd(&3.);
        let e = f64x4::splat(3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_broadcast_ps() {
        let a = f32x4::new(4., 3., 2., 5.);
        let r = avx::_mm256_broadcast_ps(&a);
        let e = f32x8::new(4., 3., 2., 5., 4., 3., 2., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_broadcast_pd() {
        let a = f64x2::new(4., 3.);
        let r = avx::_mm256_broadcast_pd(&a);
        let e = f64x4::new(4., 3., 4., 3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insertf128_ps() {
        let a = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        let b = f32x4::new(4., 9., 16., 25.);
        let r = avx::_mm256_insertf128_ps(a, b, 0);
        let e = f32x8::new(4., 9., 16., 25., 8., 9., 64., 50.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insertf128_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x2::new(5., 6.);
        let r = avx::_mm256_insertf128_pd(a, b, 0);
        let e = f64x4::new(5., 6., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insertf128_si256() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let b = __m128i::from(i64x2::new(5, 6));
        let r = avx::_mm256_insertf128_si256(a, b, 0);
        let e = i64x4::new(5, 6, 3, 4);
        assert_eq!(r, __m256i::from(e));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insert_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = avx::_mm256_insert_epi8(a, 0, 31);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 0,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insert_epi16() {
        let a =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx::_mm256_insert_epi16(a, 0, 15);
        let e =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insert_epi32() {
        let a = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx::_mm256_insert_epi32(a, 0, 7);
        let e = i32x8::new(1, 2, 3, 4, 5, 6, 7, 0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_insert_epi64() {
        let a = i64x4::new(1, 2, 3, 4);
        let r = avx::_mm256_insert_epi64(a, 0, 3);
        let e = i64x4::new(1, 2, 3, 0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_load_pd() {
        let a = avx::_mm256_setr_pd(1., 2., 3., 4.);
        let p = &a as *const _ as *const f64;
        let r = avx::_mm256_load_pd(p);
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_store_pd() {
        let a = avx::_mm256_setr_pd(1., 2., 3., 4.);
        let mut r = avx::_mm256_undefined_pd();
        avx::_mm256_store_pd(&mut r as *mut _ as *mut f64, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_load_ps() {
        let a = avx::_mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let p = &a as *const _ as *const f32;
        let r = avx::_mm256_load_ps(p);
        let e = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_store_ps() {
        let a = avx::_mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let mut r = avx::_mm256_undefined_ps();
        avx::_mm256_store_ps(&mut r as *mut _ as *mut f32, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu_pd() {
        let a = &[1.0f64, 2., 3., 4.];
        let p = a.as_ptr();
        let r = avx::_mm256_loadu_pd(black_box(p));
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu_pd() {
        let a = f64x4::splat(9.);
        let mut r = avx::_mm256_undefined_pd();
        avx::_mm256_storeu_pd(&mut r as *mut _ as *mut f64, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu_ps() {
        let a = &[4., 3., 2., 5., 8., 9., 64., 50.];
        let p = a.as_ptr();
        let r = avx::_mm256_loadu_ps(black_box(p));
        let e = f32x8::new(4., 3., 2., 5., 8., 9., 64., 50.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu_ps() {
        let a = f32x8::splat(9.);
        let mut r = avx::_mm256_undefined_ps();
        avx::_mm256_storeu_ps(&mut r as *mut _ as *mut f32, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_load_si256() {
        let a = __m256i::from(avx::_mm256_setr_epi64x(1, 2, 3, 4));
        let p = &a as *const _;
        let r = avx::_mm256_load_si256(p);
        let e = i64x4::new(1, 2, 3, 4);
        assert_eq!(r, __m256i::from(e));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_store_si256() {
        let a = __m256i::from(avx::_mm256_setr_epi64x(1, 2, 3, 4));
        let mut r = avx::_mm256_undefined_si256();
        avx::_mm256_store_si256(&mut r as *mut _, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu_si256() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let p = &a as *const _;
        let r = avx::_mm256_loadu_si256(black_box(p));
        let e = i64x4::new(1, 2, 3, 4);
        assert_eq!(r, __m256i::from(e));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu_si256() {
        let a = i8x32::splat(9);
        let mut r = avx::_mm256_undefined_si256();
        avx::_mm256_storeu_si256(&mut r as *mut _, a);
        assert_eq!(r, a);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_maskload_pd() {
        let a = &[1.0f64, 2., 3., 4.];
        let p = a.as_ptr();
        let mask = i64x4::new(0, !0, 0, !0);
        let r = avx::_mm256_maskload_pd(black_box(p), mask);
        let e = f64x4::new(0., 2., 0., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_maskstore_pd() {
        let mut r = f64x4::splat(0.);
        let mask = i64x4::new(0, !0, 0, !0);
        let a = f64x4::new(1., 2., 3., 4.);
        avx::_mm256_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = f64x4::new(0., 2., 0., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_maskload_pd() {
        let a = &[1.0f64, 2.];
        let p = a.as_ptr();
        let mask = i64x2::new(0, !0);
        let r = avx::_mm_maskload_pd(black_box(p), mask);
        let e = f64x2::new(0., 2.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_maskstore_pd() {
        let mut r = f64x2::splat(0.);
        let mask = i64x2::new(0, !0);
        let a = f64x2::new(1., 2.);
        avx::_mm_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = f64x2::new(0., 2.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4., 5., 6., 7., 8.];
        let p = a.as_ptr();
        let mask = i32x8::new(0, !0, 0, !0, 0, !0, 0, !0);
        let r = avx::_mm256_maskload_ps(black_box(p), mask);
        let e = f32x8::new(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_maskstore_ps() {
        let mut r = f32x8::splat(0.);
        let mask = i32x8::new(0, !0, 0, !0, 0, !0, 0, !0);
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        avx::_mm256_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = f32x8::new(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4.];
        let p = a.as_ptr();
        let mask = i32x4::new(0, !0, 0, !0);
        let r = avx::_mm_maskload_ps(black_box(p), mask);
        let e = f32x4::new(0., 2., 0., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_maskstore_ps() {
        let mut r = f32x4::splat(0.);
        let mask = i32x4::new(0, !0, 0, !0);
        let a = f32x4::new(1., 2., 3., 4.);
        avx::_mm_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = f32x4::new(0., 2., 0., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_movehdup_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_movehdup_ps(a);
        let e = f32x8::new(2., 2., 4., 4., 6., 6., 8., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_moveldup_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_moveldup_ps(a);
        let e = f32x8::new(1., 1., 3., 3., 5., 5., 7., 7.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_movedup_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_movedup_pd(a);
        let e = f64x4::new(1., 1., 3., 3.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_lddqu_si256() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let p = &a as *const _;
        let r = avx::_mm256_lddqu_si256(black_box(p));
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_rcp_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_rcp_ps(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = f32x8::new(
            0.99975586, 0.49987793, 0.33325195, 0.24993896,
            0.19995117, 0.16662598, 0.14282227, 0.12496948,
        );
        let rel_err = 0.00048828125;
        for i in 0..8 {
            assert_approx_eq!(r.extract(i), e.extract(i), 2. * rel_err);
        }
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_rsqrt_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_rsqrt_ps(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = f32x8::new(
            0.99975586, 0.7069092, 0.5772705, 0.49987793,
            0.44714355, 0.40820313, 0.3779297, 0.3534546,
        );
        let rel_err = 0.00048828125;
        for i in 0..8 {
            assert_approx_eq!(r.extract(i), e.extract(i), 2. * rel_err);
        }
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_unpackhi_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_unpackhi_pd(a, b);
        let e = f64x4::new(2., 6., 4., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_unpackhi_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = f32x8::new(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = avx::_mm256_unpackhi_ps(a, b);
        let e = f32x8::new(3., 11., 4., 12., 7., 15., 8., 16.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_unpacklo_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_unpacklo_pd(a, b);
        let e = f64x4::new(1., 5., 3., 7.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_unpacklo_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = f32x8::new(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = avx::_mm256_unpacklo_ps(a, b);
        let e = f32x8::new(1., 9., 2., 10., 5., 13., 6., 14.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testz_si256() {
        let a = i64x4::new(1, 2, 3, 4);
        let b = i64x4::new(5, 6, 7, 8);
        let r = avx::_mm256_testz_si256(a, b);
        assert_eq!(r, 0);
        let b = i64x4::splat(0);
        let r = avx::_mm256_testz_si256(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testc_si256() {
        let a = i64x4::new(1, 2, 3, 4);
        let b = i64x4::new(5, 6, 7, 8);
        let r = avx::_mm256_testc_si256(a, b);
        assert_eq!(r, 0);
        let b = i64x4::splat(0);
        let r = avx::_mm256_testc_si256(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testnzc_si256() {
        let a = i64x4::new(1, 2, 3, 4);
        let b = i64x4::new(5, 6, 7, 8);
        let r = avx::_mm256_testnzc_si256(a, b);
        assert_eq!(r, 1);
        let a = i64x4::new(0, 0, 0, 0);
        let b = i64x4::new(0, 0, 0, 0);
        let r = avx::_mm256_testnzc_si256(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testz_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = f64x4::splat(-1.);
        let r = avx::_mm256_testz_pd(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testc_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = f64x4::splat(1.);
        let b = f64x4::splat(-1.);
        let r = avx::_mm256_testc_pd(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testnzc_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let b = f64x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = f64x4::new(1., -1., -1., -1.);
        let b = f64x4::new(-1., -1., 1., 1.);
        let r = avx::_mm256_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testz_pd() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(5., 6.);
        let r = avx::_mm_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = f64x2::splat(-1.);
        let r = avx::_mm_testz_pd(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testc_pd() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(5., 6.);
        let r = avx::_mm_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = f64x2::splat(1.);
        let b = f64x2::splat(-1.);
        let r = avx::_mm_testc_pd(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testnzc_pd() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(5., 6.);
        let r = avx::_mm_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = f64x2::new(1., -1.);
        let b = f64x2::new(-1., -1.);
        let r = avx::_mm_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testz_ps() {
        let a = f32x8::splat(1.);
        let r = avx::_mm256_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = f32x8::splat(-1.);
        let r = avx::_mm256_testz_ps(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testc_ps() {
        let a = f32x8::splat(1.);
        let r = avx::_mm256_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = f32x8::splat(-1.);
        let r = avx::_mm256_testc_ps(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_testnzc_ps() {
        let a = f32x8::splat(1.);
        let r = avx::_mm256_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = f32x8::new(1., -1., -1., -1., -1., -1., -1., -1.);
        let b = f32x8::new(-1., -1., 1., 1., 1., 1., 1., 1.);
        let r = avx::_mm256_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testz_ps() {
        let a = f32x4::splat(1.);
        let r = avx::_mm_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = f32x4::splat(-1.);
        let r = avx::_mm_testz_ps(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testc_ps() {
        let a = f32x4::splat(1.);
        let r = avx::_mm_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = f32x4::splat(-1.);
        let r = avx::_mm_testc_ps(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test = "avx"]
    unsafe fn _mm_testnzc_ps() {
        let a = f32x4::splat(1.);
        let r = avx::_mm_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = f32x4::new(1., -1., -1., -1.);
        let b = f32x4::new(-1., -1., 1., 1.);
        let r = avx::_mm_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_movemask_pd() {
        let a = f64x4::new(1., -2., 3., -4.);
        let r = avx::_mm256_movemask_pd(a);
        assert_eq!(r, 0xA);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_movemask_ps() {
        let a = f32x8::new(1., -2., 3., -4., 1., -2., 3., -4.);
        let r = avx::_mm256_movemask_ps(a);
        assert_eq!(r, 0xAA);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setzero_pd() {
        let r = avx::_mm256_setzero_pd();
        assert_eq!(r, f64x4::splat(0.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setzero_ps() {
        let r = avx::_mm256_setzero_ps();
        assert_eq!(r, f32x8::splat(0.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setzero_si256() {
        let r = avx::_mm256_setzero_si256();
        assert_eq!(r, i8x32::splat(0));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_pd() {
        let r = avx::_mm256_set_pd(1., 2., 3., 4.);
        assert_eq!(r, f64x4::new(4., 3., 2., 1.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_ps() {
        let r = avx::_mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, f32x8::new(8., 7., 6., 5., 4., 3., 2., 1.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = avx::_mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            32, 31, 30, 29, 28, 27, 26, 25,
            24, 23, 22, 21, 20, 19, 18, 17,
            16, 15, 14, 13, 12, 11, 10, 9,
            8, 7, 6, 5, 4, 3, 2, 1
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = avx::_mm256_set_epi16(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        assert_eq!(
            r,
            i16x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        );
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_epi32() {
        let r = avx::_mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(r, i32x8::new(8, 7, 6, 5, 4, 3, 2, 1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_epi64x() {
        let r = avx::_mm256_set_epi64x(1, 2, 3, 4);
        assert_eq!(r, i64x4::new(4, 3, 2, 1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_pd() {
        let r = avx::_mm256_setr_pd(1., 2., 3., 4.);
        assert_eq!(r, f64x4::new(1., 2., 3., 4.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_ps() {
        let r = avx::_mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = avx::_mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );

        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = avx::_mm256_setr_epi16(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        assert_eq!(
            r,
            i16x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        );
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_epi32() {
        let r = avx::_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(r, i32x8::new(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_epi64x() {
        let r = avx::_mm256_setr_epi64x(1, 2, 3, 4);
        assert_eq!(r, i64x4::new(1, 2, 3, 4));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_pd() {
        let r = avx::_mm256_set1_pd(1.);
        assert_eq!(r, f64x4::splat(1.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_ps() {
        let r = avx::_mm256_set1_ps(1.);
        assert_eq!(r, f32x8::splat(1.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_epi8() {
        let r = avx::_mm256_set1_epi8(1);
        assert_eq!(r, i8x32::splat(1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_epi16() {
        let r = avx::_mm256_set1_epi16(1);
        assert_eq!(r, i16x16::splat(1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_epi32() {
        let r = avx::_mm256_set1_epi32(1);
        assert_eq!(r, i32x8::splat(1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set1_epi64x() {
        let r = avx::_mm256_set1_epi64x(1);
        assert_eq!(r, i64x4::splat(1));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castpd_ps() {
        let a = f64x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_castpd_ps(a);
        let e = f32x8::new(0., 1.875, 0., 2., 0., 2.125, 0., 2.25);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castps_pd() {
        let a = f32x8::new(0., 1.875, 0., 2., 0., 2.125, 0., 2.25);
        let r = avx::_mm256_castps_pd(a);
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castps_si256() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_castps_si256(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            0, 0, -128, 63, 0, 0, 0, 64,
            0, 0, 64, 64, 0, 0, -128, 64,
            0, 0, -96, 64, 0, 0, -64, 64,
            0, 0, -32, 64, 0, 0, 0, 65,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castsi256_ps() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            0, 0, -128, 63, 0, 0, 0, 64,
            0, 0, 64, 64, 0, 0, -128, 64,
            0, 0, -96, 64, 0, 0, -64, 64,
            0, 0, -32, 64, 0, 0, 0, 65,
        );
        let r = avx::_mm256_castsi256_ps(a);
        let e = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castpd_si256() {
        let a = f64x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_castpd_si256(a);
        assert_eq!(r, __m256i::from(i64x4::new(1, 2, 3, 4)));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castsi256_pd() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let r = avx::_mm256_castsi256_pd(a);
        assert_eq!(r, f64x4::new(1., 2., 3., 4.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castps256_ps128() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = avx::_mm256_castps256_ps128(a);
        assert_eq!(r, f32x4::new(1., 2., 3., 4.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castpd256_pd128() {
        let a = f64x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_castpd256_pd128(a);
        assert_eq!(r, f64x2::new(1., 2.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_castsi256_si128() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let r = avx::_mm256_castsi256_si128(a);
        assert_eq!(r, __m128i::from(i64x2::new(1, 2)));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_zextps128_ps256() {
        let a = f32x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_zextps128_ps256(a);
        let e = f32x8::new(1., 2., 3., 4., 0., 0., 0., 0.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_zextsi128_si256() {
        let a = __m128i::from(i64x2::new(1, 2));
        let r = avx::_mm256_zextsi128_si256(a);
        let e = __m256i::from(i64x4::new(1, 2, 0, 0));
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_zextpd128_pd256() {
        let a = f64x2::new(1., 2.);
        let r = avx::_mm256_zextpd128_pd256(a);
        let e = f64x4::new(1., 2., 0., 0.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_m128() {
        let hi = f32x4::new(5., 6., 7., 8.);
        let lo = f32x4::new(1., 2., 3., 4.);
        let r = avx::_mm256_set_m128(hi, lo);
        let e = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_m128d() {
        let hi = f64x2::new(3., 4.);
        let lo = f64x2::new(1., 2.);
        let r = avx::_mm256_set_m128d(hi, lo);
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_set_m128i() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let hi = i8x16::new(
            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32,
        );
        let lo =
            i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = avx::_mm256_set_m128i(hi, lo);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_m128() {
        let lo = f32x4::new(1., 2., 3., 4.);
        let hi = f32x4::new(5., 6., 7., 8.);
        let r = avx::_mm256_setr_m128(lo, hi);
        let e = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_m128d() {
        let lo = f64x2::new(1., 2.);
        let hi = f64x2::new(3., 4.);
        let r = avx::_mm256_setr_m128d(lo, hi);
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_setr_m128i() {
        let lo =
            i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let hi = i8x16::new(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = avx::_mm256_setr_m128i(lo, hi);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu2_m128() {
        let hi = &[5., 6., 7., 8.];
        let hiaddr = hi.as_ptr();
        let lo = &[1., 2., 3., 4.];
        let loaddr = lo.as_ptr();
        let r = avx::_mm256_loadu2_m128(hiaddr, loaddr);
        let e = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu2_m128d() {
        let hi = &[3., 4.];
        let hiaddr = hi.as_ptr();
        let lo = &[1., 2.];
        let loaddr = lo.as_ptr();
        let r = avx::_mm256_loadu2_m128d(hiaddr, loaddr);
        let e = f64x4::new(1., 2., 3., 4.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_loadu2_m128i() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let hi = i8x16::new(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let lo =
            i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = avx::_mm256_loadu2_m128i(
            &hi as *const _ as *const _,
            &lo as *const _ as *const _,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu2_m128() {
        use x86::i586::sse::_mm_undefined_ps;
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let mut hi = _mm_undefined_ps();
        let mut lo = _mm_undefined_ps();
        avx::_mm256_storeu2_m128(
            &mut hi as *mut _ as *mut f32,
            &mut lo as *mut _ as *mut f32,
            a,
        );
        assert_eq!(hi, f32x4::new(5., 6., 7., 8.));
        assert_eq!(lo, f32x4::new(1., 2., 3., 4.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu2_m128d() {
        use x86::i586::sse2::_mm_undefined_pd;
        let a = f64x4::new(1., 2., 3., 4.);
        let mut hi = _mm_undefined_pd();
        let mut lo = _mm_undefined_pd();
        avx::_mm256_storeu2_m128d(
            &mut hi as *mut _ as *mut f64,
            &mut lo as *mut _ as *mut f64,
            a,
        );
        assert_eq!(hi, f64x2::new(3., 4.));
        assert_eq!(lo, f64x2::new(1., 2.));
    }

    #[simd_test = "avx"]
    unsafe fn _mm256_storeu2_m128i() {
        use x86::i586::sse2::_mm_undefined_si128;
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let mut hi = _mm_undefined_si128();
        let mut lo = _mm_undefined_si128();
        avx::_mm256_storeu2_m128i(&mut hi as *mut _, &mut lo as *mut _, a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x16::new(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );

        assert_eq!(hi, e);
        assert_eq!(
            lo,
            i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        );
    }
}
