//! LoongArch64 LASX intrinsics - intrinsics::simd implementation

use super::super::{simd::*, *};
use crate::core_arch::simd::*;
use crate::intrinsics::simd::*;
use crate::mem::transmute;

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            0, 2, 4, 6, 8, 10, 12, 14, 32, 34, 36, 38, 40, 42, 44, 46,
            16, 18, 20, 22, 24, 26, 28, 30, 48, 50, 52, 54, 56, 58, 60, 62
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2, 8, 10, 4, 6, 12, 14])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 4, 2, 6])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            1, 3, 5, 7, 9, 11, 13, 15, 33, 35, 37, 39, 41, 43, 45, 47,
            17, 19, 21, 23, 25, 27, 29, 31, 49, 51, 53, 55, 57, 59, 61, 63
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3, 5, 7, 17, 19, 21, 23, 9, 11, 13, 15, 25, 27, 29, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3, 9, 11, 5, 7, 13, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 5, 3, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
            24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [2, 10, 3, 11, 6, 14, 7, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 5, 3, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
            16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 8, 1, 9, 4, 12, 5, 13])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 4, 2, 6])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_b<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
            I + 16, I + 16, I + 16, I + 16, I + 16, I + 16, I + 16, I + 16,
            I + 16, I + 16, I + 16, I + 16, I + 16, I + 16, I + 16, I + 16
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_h<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            I, I, I, I, I, I, I, I,
            I + 8, I + 8, I + 8, I + 8, I + 8, I + 8, I + 8, I + 8
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_w<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I, I, I, I + 4, I + 4, I + 4, I + 4])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_d<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I, I + 2, I + 2])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replve0_b<T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replve0_h<T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replve0_w<T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [0, 0, 0, 0, 0, 0, 0, 0])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replve0_d<T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [0, 0, 0, 0])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replve0_q<T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [0, 1, 0, 1])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            0, 32, 2, 34, 4, 36, 6, 38, 8, 40, 10, 42, 12, 44, 14, 46,
            16, 48, 18, 50, 20, 52, 22, 54, 24, 56, 26, 58, 28, 60, 30, 62
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 4, 2, 6])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(
        b,
        a,
        [
            1, 33, 3, 35, 5, 37, 7, 39, 9, 41, 11, 43, 13, 45, 15, 47,
            17, 49, 19, 51, 21, 53, 23, 55, 25, 57, 27, 59, 29, 61, 31, 63
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 5, 3, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_shuf4i_b<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            ((I >> 0) & 3) + 0, ((I >> 2) & 3) + 0, ((I >> 4) & 3) + 0, ((I >> 6) & 3) + 0,
            ((I >> 0) & 3) + 4, ((I >> 2) & 3) + 4, ((I >> 4) & 3) + 4, ((I >> 6) & 3) + 4,
            ((I >> 0) & 3) + 8, ((I >> 2) & 3) + 8, ((I >> 4) & 3) + 8, ((I >> 6) & 3) + 8,
            ((I >> 0) & 3) + 12, ((I >> 2) & 3) + 12, ((I >> 4) & 3) + 12, ((I >> 6) & 3) + 12,
            ((I >> 0) & 3) + 16, ((I >> 2) & 3) + 16, ((I >> 4) & 3) + 16, ((I >> 6) & 3) + 16,
            ((I >> 0) & 3) + 20, ((I >> 2) & 3) + 20, ((I >> 4) & 3) + 20, ((I >> 6) & 3) + 20,
            ((I >> 0) & 3) + 24, ((I >> 2) & 3) + 24, ((I >> 4) & 3) + 24, ((I >> 6) & 3) + 24,
            ((I >> 0) & 3) + 28, ((I >> 2) & 3) + 28, ((I >> 4) & 3) + 28, ((I >> 6) & 3) + 28
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_shuf4i_h<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            ((I >> 0) & 3) + 0, ((I >> 2) & 3) + 0, ((I >> 4) & 3) + 0, ((I >> 6) & 3) + 0,
            ((I >> 0) & 3) + 4, ((I >> 2) & 3) + 4, ((I >> 4) & 3) + 4, ((I >> 6) & 3) + 4,
            ((I >> 0) & 3) + 8, ((I >> 2) & 3) + 8, ((I >> 4) & 3) + 8, ((I >> 6) & 3) + 8,
            ((I >> 0) & 3) + 12, ((I >> 2) & 3) + 12, ((I >> 4) & 3) + 12, ((I >> 6) & 3) + 12,
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_shuf4i_w<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(
        a,
        a,
        [
            ((I >> 0) & 3) + 0, ((I >> 2) & 3) + 0, ((I >> 4) & 3) + 0, ((I >> 6) & 3) + 0,
            ((I >> 0) & 3) + 4, ((I >> 2) & 3) + 4, ((I >> 4) & 3) + 4, ((I >> 6) & 3) + 4
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bsll<const I: u32, T: Copy + const SimdExt>(a: T) -> T {
    let z = simd_ext_splat(0);
    match I & 0xf {
        0 => simd_shuffle!(
            a,
            z,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
            ]
        ),
        1 => simd_shuffle!(
            a,
            z,
            [
                32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ]
        ),
        2 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            ]
        ),
        3 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ]
        ),
        4 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            ]
        ),
        5 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            ]
        ),
        6 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            ]
        ),
        7 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            ]
        ),
        8 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6, 7,
                32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22, 23,
            ]
        ),
        9 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5, 6,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21, 22,
            ]
        ),
        10 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4, 5,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20, 21,
            ]
        ),
        11 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3, 4,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19, 20,
            ]
        ),
        12 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2, 3,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18, 19,
            ]
        ),
        13 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1, 2,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17, 18,
            ]
        ),
        14 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 1,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 17,
            ]
        ),
        15 => simd_shuffle!(
            a,
            z,
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16,
            ]
        ),
        _ => unreachable!(),
    }
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bsrl<const I: u32, T: Copy + const SimdExt>(a: T) -> T {
    let z = simd_ext_splat(0);
    match I & 0xf {
        0 => simd_shuffle!(
            a,
            z,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
            ]
        ),
        1 => simd_shuffle!(
            a,
            z,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        ),
        2 => simd_shuffle!(
            a,
            z,
            [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32
            ]
        ),
        3 => simd_shuffle!(
            a,
            z,
            [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32
            ]
        ),
        4 => simd_shuffle!(
            a,
            z,
            [
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32
            ]
        ),
        5 => simd_shuffle!(
            a,
            z,
            [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32
            ]
        ),
        6 => simd_shuffle!(
            a,
            z,
            [
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32
            ]
        ),
        7 => simd_shuffle!(
            a,
            z,
            [
                7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        8 => simd_shuffle!(
            a,
            z,
            [
                8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        9 => simd_shuffle!(
            a,
            z,
            [
                9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        10 => simd_shuffle!(
            a,
            z,
            [
                10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        11 => simd_shuffle!(
            a,
            z,
            [
                11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        12 => simd_shuffle!(
            a,
            z,
            [
                12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        13 => simd_shuffle!(
            a,
            z,
            [
                13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        14 => simd_shuffle!(
            a,
            z,
            [
                14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        15 => simd_shuffle!(
            a,
            z,
            [
                15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
            ]
        ),
        _ => unreachable!(),
    }
}

impl_vv!("lasx", lasx_xvpcnt_b, simd_ctpop, m256i, i8x32);
impl_vv!("lasx", lasx_xvpcnt_h, simd_ctpop, m256i, i16x16);
impl_vv!("lasx", lasx_xvpcnt_w, simd_ctpop, m256i, i32x8);
impl_vv!("lasx", lasx_xvpcnt_d, simd_ctpop, m256i, i64x4);
impl_vv!("lasx", lasx_xvclz_b, simd_ctlz, m256i, i8x32);
impl_vv!("lasx", lasx_xvclz_h, simd_ctlz, m256i, i16x16);
impl_vv!("lasx", lasx_xvclz_w, simd_ctlz, m256i, i32x8);
impl_vv!("lasx", lasx_xvclz_d, simd_ctlz, m256i, i64x4);
impl_vv!("lasx", lasx_xvneg_b, simd_neg, m256i, i8x32);
impl_vv!("lasx", lasx_xvneg_h, simd_neg, m256i, i16x16);
impl_vv!("lasx", lasx_xvneg_w, simd_neg, m256i, i32x8);
impl_vv!("lasx", lasx_xvneg_d, simd_neg, m256i, i64x4);
impl_vv!("lasx", lasx_xvfsqrt_s, simd_fsqrt, m256, f32x8);
impl_vv!("lasx", lasx_xvfsqrt_d, simd_fsqrt, m256d, f64x4);
impl_vv!("lasx", lasx_xvfrsqrt_s, simd_ext_frsqrt_s, m256, f32x8);
impl_vv!("lasx", lasx_xvfrsqrt_d, simd_ext_frsqrt_d, m256d, f64x4);
impl_vv!("lasx", lasx_xvfrecip_s, simd_ext_frecip_s, m256, f32x8);
impl_vv!("lasx", lasx_xvfrecip_d, simd_ext_frecip_d, m256d, f64x4);
impl_vv!("lasx", lasx_xvfrintrp_s, simd_ceil, m256, f32x8);
impl_vv!("lasx", lasx_xvfrintrp_d, simd_ceil, m256d, f64x4);
impl_vv!("lasx", lasx_xvfrintrm_s, simd_floor, m256, f32x8);
impl_vv!("lasx", lasx_xvfrintrm_d, simd_floor, m256d, f64x4);
impl_vv!("lasx", lasx_xvfrintrz_s, simd_trunc, m256, f32x8);
impl_vv!("lasx", lasx_xvfrintrz_d, simd_trunc, m256d, f64x4);
impl_vv!("lasx", lasx_xvreplve0_b, simd_ext_replve0_b, m256i, i8x32);
impl_vv!("lasx", lasx_xvreplve0_h, simd_ext_replve0_h, m256i, i16x16);
impl_vv!("lasx", lasx_xvreplve0_w, simd_ext_replve0_w, m256i, i32x8);
impl_vv!("lasx", lasx_xvreplve0_d, simd_ext_replve0_d, m256i, i64x4);
impl_vv!("lasx", lasx_xvreplve0_q, simd_ext_replve0_q, m256i, i64x4);

impl_gv!("lasx", lasx_xvreplgr2vr_b, simd_ext_splat, m256i, i8x32, i32);
impl_gv!("lasx", lasx_xvreplgr2vr_h, simd_ext_splat, m256i, i16x16, i32);
impl_gv!("lasx", lasx_xvreplgr2vr_w, simd_ext_splat, m256i, i32x8, i32);
impl_gv!("lasx", lasx_xvreplgr2vr_d, simd_ext_splat, m256i, i64x4, i64);

impl_ggv!("lasx", lasx_xvldx, simd_ext_ldx, m256i, i8x32, *const i8, i64, unsafe);

impl_gsv!("lasx", lasx_xvld, simd_ext_ld, m256i, i8x32, *const i8, 12, const, unsafe);

impl_sv!("lasx", lasx_xvrepli_b, simd_ext_splat, m256i, i8x32, 10);
impl_sv!("lasx", lasx_xvrepli_h, simd_ext_splat, m256i, i16x16, 10);
impl_sv!("lasx", lasx_xvrepli_w, simd_ext_splat, m256i, i32x8, 10);
impl_sv!("lasx", lasx_xvrepli_d, simd_ext_splat, m256i, i64x4, 10);
impl_sv!("lasx", lasx_xvldi, simd_ext_ldi, m256i, i64x4, 13, const);

impl_vvv!("lasx", lasx_xvadd_b, simd_add, m256i, i8x32);
impl_vvv!("lasx", lasx_xvadd_h, simd_add, m256i, i16x16);
impl_vvv!("lasx", lasx_xvadd_w, simd_add, m256i, i32x8);
impl_vvv!("lasx", lasx_xvadd_d, simd_add, m256i, i64x4);
impl_vvv!("lasx", lasx_xvsub_b, simd_sub, m256i, i8x32);
impl_vvv!("lasx", lasx_xvsub_h, simd_sub, m256i, i16x16);
impl_vvv!("lasx", lasx_xvsub_w, simd_sub, m256i, i32x8);
impl_vvv!("lasx", lasx_xvsub_d, simd_sub, m256i, i64x4);
impl_vvv!("lasx", lasx_xvmax_b, simd_imax, m256i, i8x32);
impl_vvv!("lasx", lasx_xvmax_h, simd_imax, m256i, i16x16);
impl_vvv!("lasx", lasx_xvmax_w, simd_imax, m256i, i32x8);
impl_vvv!("lasx", lasx_xvmax_d, simd_imax, m256i, i64x4);
impl_vvv!("lasx", lasx_xvmax_bu, simd_imax, m256i, u8x32);
impl_vvv!("lasx", lasx_xvmax_hu, simd_imax, m256i, u16x16);
impl_vvv!("lasx", lasx_xvmax_wu, simd_imax, m256i, u32x8);
impl_vvv!("lasx", lasx_xvmax_du, simd_imax, m256i, u64x4);
impl_vvv!("lasx", lasx_xvmin_b, simd_imin, m256i, i8x32);
impl_vvv!("lasx", lasx_xvmin_h, simd_imin, m256i, i16x16);
impl_vvv!("lasx", lasx_xvmin_w, simd_imin, m256i, i32x8);
impl_vvv!("lasx", lasx_xvmin_d, simd_imin, m256i, i64x4);
impl_vvv!("lasx", lasx_xvmin_bu, simd_imin, m256i, u8x32);
impl_vvv!("lasx", lasx_xvmin_hu, simd_imin, m256i, u16x16);
impl_vvv!("lasx", lasx_xvmin_wu, simd_imin, m256i, u32x8);
impl_vvv!("lasx", lasx_xvmin_du, simd_imin, m256i, u64x4);
impl_vvv!("lasx", lasx_xvseq_b, simd_eq, m256i, i8x32);
impl_vvv!("lasx", lasx_xvseq_h, simd_eq, m256i, i16x16);
impl_vvv!("lasx", lasx_xvseq_w, simd_eq, m256i, i32x8);
impl_vvv!("lasx", lasx_xvseq_d, simd_eq, m256i, i64x4);
impl_vvv!("lasx", lasx_xvslt_b, simd_lt, m256i, i8x32);
impl_vvv!("lasx", lasx_xvslt_h, simd_lt, m256i, i16x16);
impl_vvv!("lasx", lasx_xvslt_w, simd_lt, m256i, i32x8);
impl_vvv!("lasx", lasx_xvslt_d, simd_lt, m256i, i64x4);
impl_vvv!("lasx", lasx_xvslt_bu, simd_lt, m256i, u8x32);
impl_vvv!("lasx", lasx_xvslt_hu, simd_lt, m256i, u16x16);
impl_vvv!("lasx", lasx_xvslt_wu, simd_lt, m256i, u32x8);
impl_vvv!("lasx", lasx_xvslt_du, simd_lt, m256i, u64x4);
impl_vvv!("lasx", lasx_xvsle_b, simd_le, m256i, i8x32);
impl_vvv!("lasx", lasx_xvsle_h, simd_le, m256i, i16x16);
impl_vvv!("lasx", lasx_xvsle_w, simd_le, m256i, i32x8);
impl_vvv!("lasx", lasx_xvsle_d, simd_le, m256i, i64x4);
impl_vvv!("lasx", lasx_xvsle_bu, simd_le, m256i, u8x32);
impl_vvv!("lasx", lasx_xvsle_hu, simd_le, m256i, u16x16);
impl_vvv!("lasx", lasx_xvsle_wu, simd_le, m256i, u32x8);
impl_vvv!("lasx", lasx_xvsle_du, simd_le, m256i, u64x4);
impl_vvv!("lasx", lasx_xvmul_b, simd_mul, m256i, i8x32);
impl_vvv!("lasx", lasx_xvmul_h, simd_mul, m256i, i16x16);
impl_vvv!("lasx", lasx_xvmul_w, simd_mul, m256i, i32x8);
impl_vvv!("lasx", lasx_xvmul_d, simd_mul, m256i, i64x4);
impl_vvv!("lasx", lasx_xvdiv_b, simd_div, m256i, i8x32);
impl_vvv!("lasx", lasx_xvdiv_h, simd_div, m256i, i16x16);
impl_vvv!("lasx", lasx_xvdiv_w, simd_div, m256i, i32x8);
impl_vvv!("lasx", lasx_xvdiv_d, simd_div, m256i, i64x4);
impl_vvv!("lasx", lasx_xvdiv_bu, simd_div, m256i, u8x32);
impl_vvv!("lasx", lasx_xvdiv_hu, simd_div, m256i, u16x16);
impl_vvv!("lasx", lasx_xvdiv_wu, simd_div, m256i, u32x8);
impl_vvv!("lasx", lasx_xvdiv_du, simd_div, m256i, u64x4);
impl_vvv!("lasx", lasx_xvmod_b, simd_rem, m256i, i8x32);
impl_vvv!("lasx", lasx_xvmod_h, simd_rem, m256i, i16x16);
impl_vvv!("lasx", lasx_xvmod_w, simd_rem, m256i, i32x8);
impl_vvv!("lasx", lasx_xvmod_d, simd_rem, m256i, i64x4);
impl_vvv!("lasx", lasx_xvmod_bu, simd_rem, m256i, u8x32);
impl_vvv!("lasx", lasx_xvmod_hu, simd_rem, m256i, u16x16);
impl_vvv!("lasx", lasx_xvmod_wu, simd_rem, m256i, u32x8);
impl_vvv!("lasx", lasx_xvmod_du, simd_rem, m256i, u64x4);
impl_vvv!("lasx", lasx_xvand_v, simd_and, m256i, u8x32);
impl_vvv!("lasx", lasx_xvandn_v, simd_ext_andn, m256i, u8x32);
impl_vvv!("lasx", lasx_xvor_v, simd_or, m256i, u8x32);
impl_vvv!("lasx", lasx_xvorn_v, simd_ext_orn, m256i, u8x32);
impl_vvv!("lasx", lasx_xvnor_v, simd_ext_nor, m256i, u8x32);
impl_vvv!("lasx", lasx_xvxor_v, simd_xor, m256i, u8x32);
impl_vvv!("lasx", lasx_xvfadd_s, simd_add, m256, f32x8);
impl_vvv!("lasx", lasx_xvfadd_d, simd_add, m256d, f64x4);
impl_vvv!("lasx", lasx_xvfsub_s, simd_sub, m256, f32x8);
impl_vvv!("lasx", lasx_xvfsub_d, simd_sub, m256d, f64x4);
impl_vvv!("lasx", lasx_xvfmul_s, simd_mul, m256, f32x8);
impl_vvv!("lasx", lasx_xvfmul_d, simd_mul, m256d, f64x4);
impl_vvv!("lasx", lasx_xvfdiv_s, simd_div, m256, f32x8);
impl_vvv!("lasx", lasx_xvfdiv_d, simd_div, m256d, f64x4);
impl_vvv!("lasx", lasx_xvsll_b, simd_ext_shl, m256i, i8x32);
impl_vvv!("lasx", lasx_xvsll_h, simd_ext_shl, m256i, i16x16);
impl_vvv!("lasx", lasx_xvsll_w, simd_ext_shl, m256i, i32x8);
impl_vvv!("lasx", lasx_xvsll_d, simd_ext_shl, m256i, i64x4);
impl_vvv!("lasx", lasx_xvsra_b, simd_ext_shr, m256i, i8x32);
impl_vvv!("lasx", lasx_xvsra_h, simd_ext_shr, m256i, i16x16);
impl_vvv!("lasx", lasx_xvsra_w, simd_ext_shr, m256i, i32x8);
impl_vvv!("lasx", lasx_xvsra_d, simd_ext_shr, m256i, i64x4);
impl_vvv!("lasx", lasx_xvsrl_b, simd_ext_shr, m256i, u8x32);
impl_vvv!("lasx", lasx_xvsrl_h, simd_ext_shr, m256i, u16x16);
impl_vvv!("lasx", lasx_xvsrl_w, simd_ext_shr, m256i, u32x8);
impl_vvv!("lasx", lasx_xvsrl_d, simd_ext_shr, m256i, u64x4);
impl_vvv!("lasx", lasx_xvrotr_b, simd_ext_rotr, m256i, u8x32);
impl_vvv!("lasx", lasx_xvrotr_h, simd_ext_rotr, m256i, u16x16);
impl_vvv!("lasx", lasx_xvrotr_w, simd_ext_rotr, m256i, u32x8);
impl_vvv!("lasx", lasx_xvrotr_d, simd_ext_rotr, m256i, u64x4);
impl_vvv!("lasx", lasx_xvbitclr_b, simd_ext_bitclr, m256i, u8x32);
impl_vvv!("lasx", lasx_xvbitclr_h, simd_ext_bitclr, m256i, u16x16);
impl_vvv!("lasx", lasx_xvbitclr_w, simd_ext_bitclr, m256i, u32x8);
impl_vvv!("lasx", lasx_xvbitclr_d, simd_ext_bitclr, m256i, u64x4);
impl_vvv!("lasx", lasx_xvbitset_b, simd_ext_bitset, m256i, u8x32);
impl_vvv!("lasx", lasx_xvbitset_h, simd_ext_bitset, m256i, u16x16);
impl_vvv!("lasx", lasx_xvbitset_w, simd_ext_bitset, m256i, u32x8);
impl_vvv!("lasx", lasx_xvbitset_d, simd_ext_bitset, m256i, u64x4);
impl_vvv!("lasx", lasx_xvbitrev_b, simd_ext_bitrev, m256i, u8x32);
impl_vvv!("lasx", lasx_xvbitrev_h, simd_ext_bitrev, m256i, u16x16);
impl_vvv!("lasx", lasx_xvbitrev_w, simd_ext_bitrev, m256i, u32x8);
impl_vvv!("lasx", lasx_xvbitrev_d, simd_ext_bitrev, m256i, u64x4);
impl_vvv!("lasx", lasx_xvsadd_b, simd_saturating_add, m256i, i8x32);
impl_vvv!("lasx", lasx_xvsadd_h, simd_saturating_add, m256i, i16x16);
impl_vvv!("lasx", lasx_xvsadd_w, simd_saturating_add, m256i, i32x8);
impl_vvv!("lasx", lasx_xvsadd_d, simd_saturating_add, m256i, i64x4);
impl_vvv!("lasx", lasx_xvsadd_bu, simd_saturating_add, m256i, u8x32);
impl_vvv!("lasx", lasx_xvsadd_hu, simd_saturating_add, m256i, u16x16);
impl_vvv!("lasx", lasx_xvsadd_wu, simd_saturating_add, m256i, u32x8);
impl_vvv!("lasx", lasx_xvsadd_du, simd_saturating_add, m256i, u64x4);
impl_vvv!("lasx", lasx_xvssub_b, simd_saturating_sub, m256i, i8x32);
impl_vvv!("lasx", lasx_xvssub_h, simd_saturating_sub, m256i, i16x16);
impl_vvv!("lasx", lasx_xvssub_w, simd_saturating_sub, m256i, i32x8);
impl_vvv!("lasx", lasx_xvssub_d, simd_saturating_sub, m256i, i64x4);
impl_vvv!("lasx", lasx_xvssub_bu, simd_saturating_sub, m256i, u8x32);
impl_vvv!("lasx", lasx_xvssub_hu, simd_saturating_sub, m256i, u16x16);
impl_vvv!("lasx", lasx_xvssub_wu, simd_saturating_sub, m256i, u32x8);
impl_vvv!("lasx", lasx_xvssub_du, simd_saturating_sub, m256i, u64x4);
impl_vvv!("lasx", lasx_xvadda_b, simd_ext_adda, m256i, i8x32);
impl_vvv!("lasx", lasx_xvadda_h, simd_ext_adda, m256i, i16x16);
impl_vvv!("lasx", lasx_xvadda_w, simd_ext_adda, m256i, i32x8);
impl_vvv!("lasx", lasx_xvadda_d, simd_ext_adda, m256i, i64x4);
impl_vvv!("lasx", lasx_xvabsd_b, simd_ext_absd, m256i, i8x32);
impl_vvv!("lasx", lasx_xvabsd_h, simd_ext_absd, m256i, i16x16);
impl_vvv!("lasx", lasx_xvabsd_w, simd_ext_absd, m256i, i32x8);
impl_vvv!("lasx", lasx_xvabsd_d, simd_ext_absd, m256i, i64x4);
impl_vvv!("lasx", lasx_xvabsd_bu, simd_ext_absd, m256i, u8x32);
impl_vvv!("lasx", lasx_xvabsd_hu, simd_ext_absd, m256i, u16x16);
impl_vvv!("lasx", lasx_xvabsd_wu, simd_ext_absd, m256i, u32x8);
impl_vvv!("lasx", lasx_xvabsd_du, simd_ext_absd, m256i, u64x4);
impl_vvv!("lasx", lasx_xvmuh_b, simd_ext_muh, m256i, i8x32, i16x32);
impl_vvv!("lasx", lasx_xvmuh_h, simd_ext_muh, m256i, i16x16, i32x16);
impl_vvv!("lasx", lasx_xvmuh_w, simd_ext_muh, m256i, i32x8, i64x8);
impl_vvv!("lasx", lasx_xvmuh_d, simd_ext_muh, m256i, i64x4, i128x4);
impl_vvv!("lasx", lasx_xvmuh_bu, simd_ext_muh, m256i, u8x32, u16x32);
impl_vvv!("lasx", lasx_xvmuh_hu, simd_ext_muh, m256i, u16x16, u32x16);
impl_vvv!("lasx", lasx_xvmuh_wu, simd_ext_muh, m256i, u32x8, u64x8);
impl_vvv!("lasx", lasx_xvmuh_du, simd_ext_muh, m256i, u64x4, u128x4);
impl_vvv!("lasx", lasx_xvpickev_b, simd_ext_pickev_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvpickev_h, simd_ext_pickev_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvpickev_w, simd_ext_pickev_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvpickev_d, simd_ext_pickev_d, m256i, i64x4);
impl_vvv!("lasx", lasx_xvpickod_b, simd_ext_pickod_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvpickod_h, simd_ext_pickod_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvpickod_w, simd_ext_pickod_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvpickod_d, simd_ext_pickod_d, m256i, i64x4);
impl_vvv!("lasx", lasx_xvilvh_b, simd_ext_ilvh_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvilvh_h, simd_ext_ilvh_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvilvh_w, simd_ext_ilvh_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvilvh_d, simd_ext_ilvh_d, m256i, i64x4);
impl_vvv!("lasx", lasx_xvilvl_b, simd_ext_ilvl_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvilvl_h, simd_ext_ilvl_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvilvl_w, simd_ext_ilvl_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvilvl_d, simd_ext_ilvl_d, m256i, i64x4);
impl_vvv!("lasx", lasx_xvpackev_b, simd_ext_packev_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvpackev_h, simd_ext_packev_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvpackev_w, simd_ext_packev_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvpackev_d, simd_ext_packev_d, m256i, i64x4);
impl_vvv!("lasx", lasx_xvpackod_b, simd_ext_packod_b, m256i, i8x32);
impl_vvv!("lasx", lasx_xvpackod_h, simd_ext_packod_h, m256i, i16x16);
impl_vvv!("lasx", lasx_xvpackod_w, simd_ext_packod_w, m256i, i32x8);
impl_vvv!("lasx", lasx_xvpackod_d, simd_ext_packod_d, m256i, i64x4);

impl_vgg!("lasx", lasx_xvstx, simd_ext_stx, m256i, i8x32, *mut i8, i64, unsafe);

impl_vgs!("lasx", lasx_xvst, simd_ext_st, m256i, i8x32, *mut i8, 12, const, unsafe);

impl_vuv!("lasx", lasx_xvslli_b, simd_shl, m256i, i8x32);
impl_vuv!("lasx", lasx_xvslli_h, simd_shl, m256i, i16x16);
impl_vuv!("lasx", lasx_xvslli_w, simd_shl, m256i, i32x8);
impl_vuv!("lasx", lasx_xvslli_d, simd_shl, m256i, i64x4);
impl_vuv!("lasx", lasx_xvsrai_b, simd_shr, m256i, i8x32);
impl_vuv!("lasx", lasx_xvsrai_h, simd_shr, m256i, i16x16);
impl_vuv!("lasx", lasx_xvsrai_w, simd_shr, m256i, i32x8);
impl_vuv!("lasx", lasx_xvsrai_d, simd_shr, m256i, i64x4);
impl_vuv!("lasx", lasx_xvsrli_b, simd_shr, m256i, u8x32);
impl_vuv!("lasx", lasx_xvsrli_h, simd_shr, m256i, u16x16);
impl_vuv!("lasx", lasx_xvsrli_w, simd_shr, m256i, u32x8);
impl_vuv!("lasx", lasx_xvsrli_d, simd_shr, m256i, u64x4);
impl_vuv!("lasx", lasx_xvrotri_b, simd_ext_rotr, m256i, u8x32);
impl_vuv!("lasx", lasx_xvrotri_h, simd_ext_rotr, m256i, u16x16);
impl_vuv!("lasx", lasx_xvrotri_w, simd_ext_rotr, m256i, u32x8);
impl_vuv!("lasx", lasx_xvrotri_d, simd_ext_rotr, m256i, u64x4);
impl_vuv!("lasx", lasx_xvaddi_bu, simd_add, m256i, u8x32, 5);
impl_vuv!("lasx", lasx_xvaddi_hu, simd_add, m256i, u16x16, 5);
impl_vuv!("lasx", lasx_xvaddi_wu, simd_add, m256i, u32x8, 5);
impl_vuv!("lasx", lasx_xvaddi_du, simd_add, m256i, u64x4, 5);
impl_vuv!("lasx", lasx_xvslti_bu, simd_lt, m256i, u8x32, 5);
impl_vuv!("lasx", lasx_xvslti_hu, simd_lt, m256i, u16x16, 5);
impl_vuv!("lasx", lasx_xvslti_wu, simd_lt, m256i, u32x8, 5);
impl_vuv!("lasx", lasx_xvslti_du, simd_lt, m256i, u64x4, 5);
impl_vuv!("lasx", lasx_xvslei_bu, simd_le, m256i, u8x32, 5);
impl_vuv!("lasx", lasx_xvslei_hu, simd_le, m256i, u16x16, 5);
impl_vuv!("lasx", lasx_xvslei_wu, simd_le, m256i, u32x8, 5);
impl_vuv!("lasx", lasx_xvslei_du, simd_le, m256i, u64x4, 5);
impl_vuv!("lasx", lasx_xvmaxi_bu, simd_imax, m256i, u8x32, 5);
impl_vuv!("lasx", lasx_xvmaxi_hu, simd_imax, m256i, u16x16, 5);
impl_vuv!("lasx", lasx_xvmaxi_wu, simd_imax, m256i, u32x8, 5);
impl_vuv!("lasx", lasx_xvmaxi_du, simd_imax, m256i, u64x4, 5);
impl_vuv!("lasx", lasx_xvmini_bu, simd_imin, m256i, u8x32, 5);
impl_vuv!("lasx", lasx_xvmini_hu, simd_imin, m256i, u16x16, 5);
impl_vuv!("lasx", lasx_xvmini_wu, simd_imin, m256i, u32x8, 5);
impl_vuv!("lasx", lasx_xvmini_du, simd_imin, m256i, u64x4, 5);
impl_vuv!("lasx", lasx_xvrepl128vei_b, simd_ext_replvei_b, m256i, i8x32, 4, const);
impl_vuv!("lasx", lasx_xvrepl128vei_h, simd_ext_replvei_h, m256i, i16x16, 3, const);
impl_vuv!("lasx", lasx_xvrepl128vei_w, simd_ext_replvei_w, m256i, i32x8, 2, const);
impl_vuv!("lasx", lasx_xvrepl128vei_d, simd_ext_replvei_d, m256i, i64x4, 1, const);
impl_vuv!("lasx", lasx_xvshuf4i_b, simd_ext_shuf4i_b, m256i, i8x32, 8, const);
impl_vuv!("lasx", lasx_xvshuf4i_h, simd_ext_shuf4i_h, m256i, i16x16, 8, const);
impl_vuv!("lasx", lasx_xvshuf4i_w, simd_ext_shuf4i_w, m256i, i32x8, 8, const);
impl_vuv!("lasx", lasx_xvbsll_v, simd_ext_bsll, m256i, i8x32, 5, const);
impl_vuv!("lasx", lasx_xvbsrl_v, simd_ext_bsrl, m256i, i8x32, 5, const);

impl_vug!("lasx", lasx_xvpickve2gr_w, simd_extract, m256i, i32x8, i32, 3);
impl_vug!("lasx", lasx_xvpickve2gr_d, simd_extract, m256i, i64x4, i64, 2);
impl_vug!("lasx", lasx_xvpickve2gr_wu, simd_extract, m256i, u32x8, u32, 3);
impl_vug!("lasx", lasx_xvpickve2gr_du, simd_extract, m256i, u64x4, u64, 2);

impl_vsv!("lasx", lasx_xvseqi_b, simd_eq, m256i, i8x32, 5);
impl_vsv!("lasx", lasx_xvseqi_h, simd_eq, m256i, i16x16, 5);
impl_vsv!("lasx", lasx_xvseqi_w, simd_eq, m256i, i32x8, 5);
impl_vsv!("lasx", lasx_xvseqi_d, simd_eq, m256i, i64x4, 5);
impl_vsv!("lasx", lasx_xvslti_b, simd_lt, m256i, i8x32, 5);
impl_vsv!("lasx", lasx_xvslti_h, simd_lt, m256i, i16x16, 5);
impl_vsv!("lasx", lasx_xvslti_w, simd_lt, m256i, i32x8, 5);
impl_vsv!("lasx", lasx_xvslti_d, simd_lt, m256i, i64x4, 5);
impl_vsv!("lasx", lasx_xvslei_b, simd_le, m256i, i8x32, 5);
impl_vsv!("lasx", lasx_xvslei_h, simd_le, m256i, i16x16, 5);
impl_vsv!("lasx", lasx_xvslei_w, simd_le, m256i, i32x8, 5);
impl_vsv!("lasx", lasx_xvslei_d, simd_le, m256i, i64x4, 5);
impl_vsv!("lasx", lasx_xvmaxi_b, simd_imax, m256i, i8x32, 5);
impl_vsv!("lasx", lasx_xvmaxi_h, simd_imax, m256i, i16x16, 5);
impl_vsv!("lasx", lasx_xvmaxi_w, simd_imax, m256i, i32x8, 5);
impl_vsv!("lasx", lasx_xvmaxi_d, simd_imax, m256i, i64x4, 5);
impl_vsv!("lasx", lasx_xvmini_b, simd_imin, m256i, i8x32, 5);
impl_vsv!("lasx", lasx_xvmini_h, simd_imin, m256i, i16x16, 5);
impl_vsv!("lasx", lasx_xvmini_w, simd_imin, m256i, i32x8, 5);
impl_vsv!("lasx", lasx_xvmini_d, simd_imin, m256i, i64x4, 5);

impl_vvvv!("lasx", lasx_xvmadd_b, simd_ext_madd, m256i, i8x32);
impl_vvvv!("lasx", lasx_xvmadd_h, simd_ext_madd, m256i, i16x16);
impl_vvvv!("lasx", lasx_xvmadd_w, simd_ext_madd, m256i, i32x8);
impl_vvvv!("lasx", lasx_xvmadd_d, simd_ext_madd, m256i, i64x4);
impl_vvvv!("lasx", lasx_xvmsub_b, simd_ext_msub, m256i, i8x32);
impl_vvvv!("lasx", lasx_xvmsub_h, simd_ext_msub, m256i, i16x16);
impl_vvvv!("lasx", lasx_xvmsub_w, simd_ext_msub, m256i, i32x8);
impl_vvvv!("lasx", lasx_xvmsub_d, simd_ext_msub, m256i, i64x4);
impl_vvvv!("lasx", lasx_xvfmadd_s, simd_fma, m256, f32x8);
impl_vvvv!("lasx", lasx_xvfmadd_d, simd_fma, m256d, f64x4);
impl_vvvv!("lasx", lasx_xvfmsub_s, simd_ext_fmsub, m256, f32x8);
impl_vvvv!("lasx", lasx_xvfmsub_d, simd_ext_fmsub, m256d, f64x4);
impl_vvvv!("lasx", lasx_xvfnmadd_s, simd_ext_fnmadd, m256, f32x8);
impl_vvvv!("lasx", lasx_xvfnmadd_d, simd_ext_fnmadd, m256d, f64x4);
impl_vvvv!("lasx", lasx_xvfnmsub_s, simd_ext_fnmsub, m256, f32x8);
impl_vvvv!("lasx", lasx_xvfnmsub_d, simd_ext_fnmsub, m256d, f64x4);

impl_vugv!("lasx", lasx_xvinsgr2vr_w, simd_insert, m256i, i32x8, i32, 3);
impl_vugv!("lasx", lasx_xvinsgr2vr_d, simd_insert, m256i, i64x4, i64, 2);

#[cfg(test)]
mod tests {
    use crate::{
        core_arch::{loongarch64::*, simd::*},
        mem::transmute,
    };
    use std::hint::black_box;
    use stdarch_test::simd_test;

    #[simd_test(enable = "lasx")]
    unsafe fn xvldi() {
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<0>()));
        let r = i8x32::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        assert_eq!(r, transmute(lasx_xvldi::<255>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<1024>()));
        let r = i8x32::new(
            0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0,
            -2, 0, -2, 0, -2, 0, -2,
        );
        assert_eq!(r, transmute(lasx_xvldi::<1536>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<2048>()));
        let r = i8x32::new(
            0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1,
            -1, 0, -2, -1, -1, 0, -2, -1, -1,
        );
        assert_eq!(r, transmute(lasx_xvldi::<2560>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<3072>()));
        let r = i8x32::new(
            0, -2, -1, -1, -1, -1, -1, -1, 0, -2, -1, -1, -1, -1, -1, -1, 0, -2, -1, -1, -1, -1,
            -1, -1, 0, -2, -1, -1, -1, -1, -1, -1,
        );
        assert_eq!(r, transmute(lasx_xvldi::<3584>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-4096>()));
        let r = i8x32::new(
            -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0,
            0, -128, 0, 0, 0, -128, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3968>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3840>()));
        let r = i8x32::new(
            0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0,
            0, 0, -128, 0, 0, 0, -128, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3712>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3584>()));
        let r = i8x32::new(
            0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128,
            0, 0, 0, -128, 0, 0, 0, -128, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3456>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3328>()));
        let r = i8x32::new(
            0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0,
            -128, 0, 0, 0, -128, 0, 0, 0, -128,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3200>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-3072>()));
        let r = i8x32::new(
            -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128,
            0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2944>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2816>()));
        let r = i8x32::new(
            0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0,
            -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2688>()));
        let r = i8x32::new(
            -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
            0, -1, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2560>()));
        let r = i8x32::new(
            -1, -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0, -1,
            -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2432>()));
        let r = i8x32::new(
            -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1,
            -1, 0, 0, -1, -1, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2304>()));
        let r = i8x32::new(
            -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0,
            -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2176>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-2048>()));
        let r = i8x32::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1793>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1792>()));
        let r = i8x32::new(
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1622>()));
        let r = i8x32::new(
            0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0,
            64, 0, 0, 0, 64,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1536>()));
        let r = i8x32::new(
            0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80,
            -63, 0, 0, 80, -63, 0, 0, 80, -63,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1366>()));
        let r = i8x32::new(
            0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64,
            0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1280>()));
        let r = i8x32::new(
            0, 0, 80, -63, 0, 0, 0, 0, 0, 0, 80, -63, 0, 0, 0, 0, 0, 0, 80, -63, 0, 0, 0, 0, 0, 0,
            80, -63, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1110>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0,
            0, 0, 0, 64,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-1024>()));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 42, -64, 0, 0, 0, 0, 0, 0, 42, -64, 0, 0, 0, 0, 0, 0, 42, -64, 0, 0,
            0, 0, 0, 0, 42, -64,
        );
        assert_eq!(r, transmute(lasx_xvldi::<-854>()));
    }

    #[simd_test(enable = "lasx")]
    unsafe fn xvbsll_v() {
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<0>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<1>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<2>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 0, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<3>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<4>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<5>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<6>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<7>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20,
            21, 22, 23, 24,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<8>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 18, 19,
            20, 21, 22, 23,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<9>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 18,
            19, 20, 21, 22,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<10>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17,
            18, 19, 20, 21,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<11>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17,
            18, 19, 20,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<12>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 18, 19,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<13>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 17, 18,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<14>(black_box(transmute(a)))));
        let r = i8x32::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 17,
        );
        assert_eq!(r, transmute(lasx_xvbsll_v::<15>(black_box(transmute(a)))));
    }

    #[simd_test(enable = "lasx")]
    unsafe fn xvbsrl_v() {
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<0>(black_box(transmute(a)))));
        let r = i8x32::new(
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<1>(black_box(transmute(a)))));
        let r = i8x32::new(
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<2>(black_box(transmute(a)))));
        let r = i8x32::new(
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<3>(black_box(transmute(a)))));
        let r = i8x32::new(
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<4>(black_box(transmute(a)))));
        let r = i8x32::new(
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<5>(black_box(transmute(a)))));
        let r = i8x32::new(
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<6>(black_box(transmute(a)))));
        let r = i8x32::new(
            8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<7>(black_box(transmute(a)))));
        let r = i8x32::new(
            9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32,
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<8>(black_box(transmute(a)))));
        let r = i8x32::new(
            10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 27, 28, 29, 30, 31, 32, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<9>(black_box(transmute(a)))));
        let r = i8x32::new(
            11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 28, 29, 30, 31, 32, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<10>(black_box(transmute(a)))));
        let r = i8x32::new(
            12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 29, 30, 31, 32, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<11>(black_box(transmute(a)))));
        let r = i8x32::new(
            13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<12>(black_box(transmute(a)))));
        let r = i8x32::new(
            14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<13>(black_box(transmute(a)))));
        let r = i8x32::new(
            15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<14>(black_box(transmute(a)))));
        let r = i8x32::new(
            16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        );
        assert_eq!(r, transmute(lasx_xvbsrl_v::<15>(black_box(transmute(a)))));
    }
}
