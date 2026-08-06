//! LoongArch64 LSX intrinsics - intrinsics::simd implementation

use super::super::{simd::*, *};
use crate::core_arch::simd::*;
use crate::intrinsics::simd::*;
use crate::mem::transmute;

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2, 4, 6])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickev_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3, 5, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_pickod_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [2, 6, 3, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvh_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 4, 1, 5])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_ilvl_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_b<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_h<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I, I, I, I, I, I, I])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_w<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I, I, I])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_replvei_d<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [I, I])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 4, 2, 6])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packev_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [0, 2])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_b<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_h<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_w<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 5, 3, 7])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_packod_d<T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(b, a, [1, 3])
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
            ((I >> 0) & 3) + 12, ((I >> 2) & 3) + 12, ((I >> 4) & 3) + 12, ((I >> 6) & 3) + 12
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
            ((I >> 0) & 3) + 4, ((I >> 2) & 3) + 4, ((I >> 4) & 3) + 4, ((I >> 6) & 3) + 4
        ]
    )
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_shuf4i_w<const I: u32, T: Copy>(a: T) -> T {
    simd_shuffle!(a, a, [((I >> 0) & 3), ((I >> 2) & 3), ((I >> 4) & 3), ((I >> 6) & 3)])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
const unsafe fn simd_ext_shuf4i_d<const I: u32, T: Copy>(a: T, b: T) -> T {
    simd_shuffle!(a, b, [((I >> 0) & 3), ((I >> 2) & 3)])
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bsll<const I: u32, T: Copy + const SimdExt>(a: T) -> T {
    let z = simd_ext_splat(0);
    match I & 0xf {
        0 => simd_shuffle!(a, z, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle!(a, z, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        2 => simd_shuffle!(a, z, [16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        3 => simd_shuffle!(a, z, [16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        4 => simd_shuffle!(a, z, [16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        6 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        7 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        8 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7]),
        9 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6]),
        10 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5]),
        11 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4]),
        12 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3]),
        13 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2]),
        14 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1]),
        15 => simd_shuffle!(a, z, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0]),
        _ => unreachable!(),
    }
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bsrl<const I: u32, T: Copy + const SimdExt>(a: T) -> T {
    let z = simd_ext_splat(0);
    match I & 0xf {
        0 => simd_shuffle!(a, z, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle!(a, z, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        2 => simd_shuffle!(a, z, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16]),
        3 => simd_shuffle!(a, z, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16]),
        4 => simd_shuffle!(a, z, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16]),
        5 => simd_shuffle!(a, z, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16]),
        6 => simd_shuffle!(a, z, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16]),
        7 => simd_shuffle!(a, z, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16]),
        8 => simd_shuffle!(a, z, [8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16]),
        9 => simd_shuffle!(a, z, [9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        10 => simd_shuffle!(a, z, [10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        11 => simd_shuffle!(a, z, [11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        12 => simd_shuffle!(a, z, [12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        13 => simd_shuffle!(a, z, [13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        14 => simd_shuffle!(a, z, [14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        15 => simd_shuffle!(a, z, [15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
        _ => unreachable!(),
    }
}

impl_vv!("lsx", lsx_vpcnt_b, simd_ctpop, m128i, i8x16);
impl_vv!("lsx", lsx_vpcnt_h, simd_ctpop, m128i, i16x8);
impl_vv!("lsx", lsx_vpcnt_w, simd_ctpop, m128i, i32x4);
impl_vv!("lsx", lsx_vpcnt_d, simd_ctpop, m128i, i64x2);
impl_vv!("lsx", lsx_vclz_b, simd_ctlz, m128i, i8x16);
impl_vv!("lsx", lsx_vclz_h, simd_ctlz, m128i, i16x8);
impl_vv!("lsx", lsx_vclz_w, simd_ctlz, m128i, i32x4);
impl_vv!("lsx", lsx_vclz_d, simd_ctlz, m128i, i64x2);
impl_vv!("lsx", lsx_vneg_b, simd_neg, m128i, i8x16);
impl_vv!("lsx", lsx_vneg_h, simd_neg, m128i, i16x8);
impl_vv!("lsx", lsx_vneg_w, simd_neg, m128i, i32x4);
impl_vv!("lsx", lsx_vneg_d, simd_neg, m128i, i64x2);
impl_vv!("lsx", lsx_vfsqrt_s, simd_fsqrt, m128, f32x4);
impl_vv!("lsx", lsx_vfsqrt_d, simd_fsqrt, m128d, f64x2);
impl_vv!("lsx", lsx_vfrsqrt_s, simd_ext_frsqrt_s, m128, f32x4);
impl_vv!("lsx", lsx_vfrsqrt_d, simd_ext_frsqrt_d, m128d, f64x2);
impl_vv!("lsx", lsx_vfrecip_s, simd_ext_frecip_s, m128, f32x4);
impl_vv!("lsx", lsx_vfrecip_d, simd_ext_frecip_d, m128d, f64x2);
impl_vv!("lsx", lsx_vfrintrp_s, simd_ceil, m128, f32x4);
impl_vv!("lsx", lsx_vfrintrp_d, simd_ceil, m128d, f64x2);
impl_vv!("lsx", lsx_vfrintrm_s, simd_floor, m128, f32x4);
impl_vv!("lsx", lsx_vfrintrm_d, simd_floor, m128d, f64x2);
impl_vv!("lsx", lsx_vfrintrz_s, simd_trunc, m128, f32x4);
impl_vv!("lsx", lsx_vfrintrz_d, simd_trunc, m128d, f64x2);

impl_gv!("lsx", lsx_vreplgr2vr_b, simd_ext_splat, m128i, i8x16, i32);
impl_gv!("lsx", lsx_vreplgr2vr_h, simd_ext_splat, m128i, i16x8, i32);
impl_gv!("lsx", lsx_vreplgr2vr_w, simd_ext_splat, m128i, i32x4, i32);
impl_gv!("lsx", lsx_vreplgr2vr_d, simd_ext_splat, m128i, i64x2, i64);

impl_ggv!("lsx", lsx_vldx, simd_ext_ldx, m128i, i8x16, *const i8, i64, unsafe);

impl_gsv!("lsx", lsx_vld, simd_ext_ld, m128i, i8x16, *const i8, 12, const, unsafe);

impl_sv!("lsx", lsx_vrepli_b, simd_ext_splat, m128i, i8x16, 10);
impl_sv!("lsx", lsx_vrepli_h, simd_ext_splat, m128i, i16x8, 10);
impl_sv!("lsx", lsx_vrepli_w, simd_ext_splat, m128i, i32x4, 10);
impl_sv!("lsx", lsx_vrepli_d, simd_ext_splat, m128i, i64x2, 10);
impl_sv!("lsx", lsx_vldi, simd_ext_ldi, m128i, i64x2, 13, const);

impl_vvv!("lsx", lsx_vadd_b, simd_add, m128i, i8x16);
impl_vvv!("lsx", lsx_vadd_h, simd_add, m128i, i16x8);
impl_vvv!("lsx", lsx_vadd_w, simd_add, m128i, i32x4);
impl_vvv!("lsx", lsx_vadd_d, simd_add, m128i, i64x2);
impl_vvv!("lsx", lsx_vsub_b, simd_sub, m128i, i8x16);
impl_vvv!("lsx", lsx_vsub_h, simd_sub, m128i, i16x8);
impl_vvv!("lsx", lsx_vsub_w, simd_sub, m128i, i32x4);
impl_vvv!("lsx", lsx_vsub_d, simd_sub, m128i, i64x2);
impl_vvv!("lsx", lsx_vmax_b, simd_imax, m128i, i8x16);
impl_vvv!("lsx", lsx_vmax_h, simd_imax, m128i, i16x8);
impl_vvv!("lsx", lsx_vmax_w, simd_imax, m128i, i32x4);
impl_vvv!("lsx", lsx_vmax_d, simd_imax, m128i, i64x2);
impl_vvv!("lsx", lsx_vmax_bu, simd_imax, m128i, u8x16);
impl_vvv!("lsx", lsx_vmax_hu, simd_imax, m128i, u16x8);
impl_vvv!("lsx", lsx_vmax_wu, simd_imax, m128i, u32x4);
impl_vvv!("lsx", lsx_vmax_du, simd_imax, m128i, u64x2);
impl_vvv!("lsx", lsx_vmin_b, simd_imin, m128i, i8x16);
impl_vvv!("lsx", lsx_vmin_h, simd_imin, m128i, i16x8);
impl_vvv!("lsx", lsx_vmin_w, simd_imin, m128i, i32x4);
impl_vvv!("lsx", lsx_vmin_d, simd_imin, m128i, i64x2);
impl_vvv!("lsx", lsx_vmin_bu, simd_imin, m128i, u8x16);
impl_vvv!("lsx", lsx_vmin_hu, simd_imin, m128i, u16x8);
impl_vvv!("lsx", lsx_vmin_wu, simd_imin, m128i, u32x4);
impl_vvv!("lsx", lsx_vmin_du, simd_imin, m128i, u64x2);
impl_vvv!("lsx", lsx_vseq_b, simd_eq, m128i, i8x16);
impl_vvv!("lsx", lsx_vseq_h, simd_eq, m128i, i16x8);
impl_vvv!("lsx", lsx_vseq_w, simd_eq, m128i, i32x4);
impl_vvv!("lsx", lsx_vseq_d, simd_eq, m128i, i64x2);
impl_vvv!("lsx", lsx_vslt_b, simd_lt, m128i, i8x16);
impl_vvv!("lsx", lsx_vslt_h, simd_lt, m128i, i16x8);
impl_vvv!("lsx", lsx_vslt_w, simd_lt, m128i, i32x4);
impl_vvv!("lsx", lsx_vslt_d, simd_lt, m128i, i64x2);
impl_vvv!("lsx", lsx_vslt_bu, simd_lt, m128i, u8x16);
impl_vvv!("lsx", lsx_vslt_hu, simd_lt, m128i, u16x8);
impl_vvv!("lsx", lsx_vslt_wu, simd_lt, m128i, u32x4);
impl_vvv!("lsx", lsx_vslt_du, simd_lt, m128i, u64x2);
impl_vvv!("lsx", lsx_vsle_b, simd_le, m128i, i8x16);
impl_vvv!("lsx", lsx_vsle_h, simd_le, m128i, i16x8);
impl_vvv!("lsx", lsx_vsle_w, simd_le, m128i, i32x4);
impl_vvv!("lsx", lsx_vsle_d, simd_le, m128i, i64x2);
impl_vvv!("lsx", lsx_vsle_bu, simd_le, m128i, u8x16);
impl_vvv!("lsx", lsx_vsle_hu, simd_le, m128i, u16x8);
impl_vvv!("lsx", lsx_vsle_wu, simd_le, m128i, u32x4);
impl_vvv!("lsx", lsx_vsle_du, simd_le, m128i, u64x2);
impl_vvv!("lsx", lsx_vmul_b, simd_mul, m128i, i8x16);
impl_vvv!("lsx", lsx_vmul_h, simd_mul, m128i, i16x8);
impl_vvv!("lsx", lsx_vmul_w, simd_mul, m128i, i32x4);
impl_vvv!("lsx", lsx_vmul_d, simd_mul, m128i, i64x2);
impl_vvv!("lsx", lsx_vdiv_b, simd_div, m128i, i8x16);
impl_vvv!("lsx", lsx_vdiv_h, simd_div, m128i, i16x8);
impl_vvv!("lsx", lsx_vdiv_w, simd_div, m128i, i32x4);
impl_vvv!("lsx", lsx_vdiv_d, simd_div, m128i, i64x2);
impl_vvv!("lsx", lsx_vdiv_bu, simd_div, m128i, u8x16);
impl_vvv!("lsx", lsx_vdiv_hu, simd_div, m128i, u16x8);
impl_vvv!("lsx", lsx_vdiv_wu, simd_div, m128i, u32x4);
impl_vvv!("lsx", lsx_vdiv_du, simd_div, m128i, u64x2);
impl_vvv!("lsx", lsx_vmod_b, simd_rem, m128i, i8x16);
impl_vvv!("lsx", lsx_vmod_h, simd_rem, m128i, i16x8);
impl_vvv!("lsx", lsx_vmod_w, simd_rem, m128i, i32x4);
impl_vvv!("lsx", lsx_vmod_d, simd_rem, m128i, i64x2);
impl_vvv!("lsx", lsx_vmod_bu, simd_rem, m128i, u8x16);
impl_vvv!("lsx", lsx_vmod_hu, simd_rem, m128i, u16x8);
impl_vvv!("lsx", lsx_vmod_wu, simd_rem, m128i, u32x4);
impl_vvv!("lsx", lsx_vmod_du, simd_rem, m128i, u64x2);
impl_vvv!("lsx", lsx_vand_v, simd_and, m128i, u8x16);
impl_vvv!("lsx", lsx_vandn_v, simd_ext_andn, m128i, u8x16);
impl_vvv!("lsx", lsx_vor_v, simd_or, m128i, u8x16);
impl_vvv!("lsx", lsx_vorn_v, simd_ext_orn, m128i, u8x16);
impl_vvv!("lsx", lsx_vnor_v, simd_ext_nor, m128i, u8x16);
impl_vvv!("lsx", lsx_vxor_v, simd_xor, m128i, u8x16);
impl_vvv!("lsx", lsx_vfadd_s, simd_add, m128, f32x4);
impl_vvv!("lsx", lsx_vfadd_d, simd_add, m128d, f64x2);
impl_vvv!("lsx", lsx_vfsub_s, simd_sub, m128, f32x4);
impl_vvv!("lsx", lsx_vfsub_d, simd_sub, m128d, f64x2);
impl_vvv!("lsx", lsx_vfmul_s, simd_mul, m128, f32x4);
impl_vvv!("lsx", lsx_vfmul_d, simd_mul, m128d, f64x2);
impl_vvv!("lsx", lsx_vfdiv_s, simd_div, m128, f32x4);
impl_vvv!("lsx", lsx_vfdiv_d, simd_div, m128d, f64x2);
impl_vvv!("lsx", lsx_vsll_b, simd_ext_shl, m128i, i8x16);
impl_vvv!("lsx", lsx_vsll_h, simd_ext_shl, m128i, i16x8);
impl_vvv!("lsx", lsx_vsll_w, simd_ext_shl, m128i, i32x4);
impl_vvv!("lsx", lsx_vsll_d, simd_ext_shl, m128i, i64x2);
impl_vvv!("lsx", lsx_vsra_b, simd_ext_shr, m128i, i8x16);
impl_vvv!("lsx", lsx_vsra_h, simd_ext_shr, m128i, i16x8);
impl_vvv!("lsx", lsx_vsra_w, simd_ext_shr, m128i, i32x4);
impl_vvv!("lsx", lsx_vsra_d, simd_ext_shr, m128i, i64x2);
impl_vvv!("lsx", lsx_vsrl_b, simd_ext_shr, m128i, u8x16);
impl_vvv!("lsx", lsx_vsrl_h, simd_ext_shr, m128i, u16x8);
impl_vvv!("lsx", lsx_vsrl_w, simd_ext_shr, m128i, u32x4);
impl_vvv!("lsx", lsx_vsrl_d, simd_ext_shr, m128i, u64x2);
impl_vvv!("lsx", lsx_vrotr_b, simd_ext_rotr, m128i, u8x16);
impl_vvv!("lsx", lsx_vrotr_h, simd_ext_rotr, m128i, u16x8);
impl_vvv!("lsx", lsx_vrotr_w, simd_ext_rotr, m128i, u32x4);
impl_vvv!("lsx", lsx_vrotr_d, simd_ext_rotr, m128i, u64x2);
impl_vvv!("lsx", lsx_vbitclr_b, simd_ext_bitclr, m128i, u8x16);
impl_vvv!("lsx", lsx_vbitclr_h, simd_ext_bitclr, m128i, u16x8);
impl_vvv!("lsx", lsx_vbitclr_w, simd_ext_bitclr, m128i, u32x4);
impl_vvv!("lsx", lsx_vbitclr_d, simd_ext_bitclr, m128i, u64x2);
impl_vvv!("lsx", lsx_vbitset_b, simd_ext_bitset, m128i, u8x16);
impl_vvv!("lsx", lsx_vbitset_h, simd_ext_bitset, m128i, u16x8);
impl_vvv!("lsx", lsx_vbitset_w, simd_ext_bitset, m128i, u32x4);
impl_vvv!("lsx", lsx_vbitset_d, simd_ext_bitset, m128i, u64x2);
impl_vvv!("lsx", lsx_vbitrev_b, simd_ext_bitrev, m128i, u8x16);
impl_vvv!("lsx", lsx_vbitrev_h, simd_ext_bitrev, m128i, u16x8);
impl_vvv!("lsx", lsx_vbitrev_w, simd_ext_bitrev, m128i, u32x4);
impl_vvv!("lsx", lsx_vbitrev_d, simd_ext_bitrev, m128i, u64x2);
impl_vvv!("lsx", lsx_vsadd_b, simd_saturating_add, m128i, i8x16);
impl_vvv!("lsx", lsx_vsadd_h, simd_saturating_add, m128i, i16x8);
impl_vvv!("lsx", lsx_vsadd_w, simd_saturating_add, m128i, i32x4);
impl_vvv!("lsx", lsx_vsadd_d, simd_saturating_add, m128i, i64x2);
impl_vvv!("lsx", lsx_vsadd_bu, simd_saturating_add, m128i, u8x16);
impl_vvv!("lsx", lsx_vsadd_hu, simd_saturating_add, m128i, u16x8);
impl_vvv!("lsx", lsx_vsadd_wu, simd_saturating_add, m128i, u32x4);
impl_vvv!("lsx", lsx_vsadd_du, simd_saturating_add, m128i, u64x2);
impl_vvv!("lsx", lsx_vssub_b, simd_saturating_sub, m128i, i8x16);
impl_vvv!("lsx", lsx_vssub_h, simd_saturating_sub, m128i, i16x8);
impl_vvv!("lsx", lsx_vssub_w, simd_saturating_sub, m128i, i32x4);
impl_vvv!("lsx", lsx_vssub_d, simd_saturating_sub, m128i, i64x2);
impl_vvv!("lsx", lsx_vssub_bu, simd_saturating_sub, m128i, u8x16);
impl_vvv!("lsx", lsx_vssub_hu, simd_saturating_sub, m128i, u16x8);
impl_vvv!("lsx", lsx_vssub_wu, simd_saturating_sub, m128i, u32x4);
impl_vvv!("lsx", lsx_vssub_du, simd_saturating_sub, m128i, u64x2);
impl_vvv!("lsx", lsx_vadda_b, simd_ext_adda, m128i, i8x16);
impl_vvv!("lsx", lsx_vadda_h, simd_ext_adda, m128i, i16x8);
impl_vvv!("lsx", lsx_vadda_w, simd_ext_adda, m128i, i32x4);
impl_vvv!("lsx", lsx_vadda_d, simd_ext_adda, m128i, i64x2);
impl_vvv!("lsx", lsx_vabsd_b, simd_ext_absd, m128i, i8x16);
impl_vvv!("lsx", lsx_vabsd_h, simd_ext_absd, m128i, i16x8);
impl_vvv!("lsx", lsx_vabsd_w, simd_ext_absd, m128i, i32x4);
impl_vvv!("lsx", lsx_vabsd_d, simd_ext_absd, m128i, i64x2);
impl_vvv!("lsx", lsx_vabsd_bu, simd_ext_absd, m128i, u8x16);
impl_vvv!("lsx", lsx_vabsd_hu, simd_ext_absd, m128i, u16x8);
impl_vvv!("lsx", lsx_vabsd_wu, simd_ext_absd, m128i, u32x4);
impl_vvv!("lsx", lsx_vabsd_du, simd_ext_absd, m128i, u64x2);
impl_vvv!("lsx", lsx_vmuh_b, simd_ext_muh, m128i, i8x16, i16x16);
impl_vvv!("lsx", lsx_vmuh_h, simd_ext_muh, m128i, i16x8, i32x8);
impl_vvv!("lsx", lsx_vmuh_w, simd_ext_muh, m128i, i32x4, i64x4);
impl_vvv!("lsx", lsx_vmuh_d, simd_ext_muh, m128i, i64x2, i128x2);
impl_vvv!("lsx", lsx_vmuh_bu, simd_ext_muh, m128i, u8x16, u16x16);
impl_vvv!("lsx", lsx_vmuh_hu, simd_ext_muh, m128i, u16x8, u32x8);
impl_vvv!("lsx", lsx_vmuh_wu, simd_ext_muh, m128i, u32x4, u64x4);
impl_vvv!("lsx", lsx_vmuh_du, simd_ext_muh, m128i, u64x2, u128x2);
impl_vvv!("lsx", lsx_vpickev_b, simd_ext_pickev_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vpickev_h, simd_ext_pickev_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vpickev_w, simd_ext_pickev_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vpickev_d, simd_ext_pickev_d, m128i, i64x2);
impl_vvv!("lsx", lsx_vpickod_b, simd_ext_pickod_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vpickod_h, simd_ext_pickod_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vpickod_w, simd_ext_pickod_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vpickod_d, simd_ext_pickod_d, m128i, i64x2);
impl_vvv!("lsx", lsx_vilvh_b, simd_ext_ilvh_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vilvh_h, simd_ext_ilvh_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vilvh_w, simd_ext_ilvh_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vilvh_d, simd_ext_ilvh_d, m128i, i64x2);
impl_vvv!("lsx", lsx_vilvl_b, simd_ext_ilvl_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vilvl_h, simd_ext_ilvl_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vilvl_w, simd_ext_ilvl_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vilvl_d, simd_ext_ilvl_d, m128i, i64x2);
impl_vvv!("lsx", lsx_vpackev_b, simd_ext_packev_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vpackev_h, simd_ext_packev_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vpackev_w, simd_ext_packev_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vpackev_d, simd_ext_packev_d, m128i, i64x2);
impl_vvv!("lsx", lsx_vpackod_b, simd_ext_packod_b, m128i, i8x16);
impl_vvv!("lsx", lsx_vpackod_h, simd_ext_packod_h, m128i, i16x8);
impl_vvv!("lsx", lsx_vpackod_w, simd_ext_packod_w, m128i, i32x4);
impl_vvv!("lsx", lsx_vpackod_d, simd_ext_packod_d, m128i, i64x2);

impl_vgg!("lsx", lsx_vstx, simd_ext_stx, m128i, i8x16, *mut i8, i64, unsafe);

impl_vgs!("lsx", lsx_vst, simd_ext_st, m128i, i8x16, *mut i8, 12, const, unsafe);

impl_vuv!("lsx", lsx_vslli_b, simd_shl, m128i, i8x16);
impl_vuv!("lsx", lsx_vslli_h, simd_shl, m128i, i16x8);
impl_vuv!("lsx", lsx_vslli_w, simd_shl, m128i, i32x4);
impl_vuv!("lsx", lsx_vslli_d, simd_shl, m128i, i64x2);
impl_vuv!("lsx", lsx_vsrai_b, simd_shr, m128i, i8x16);
impl_vuv!("lsx", lsx_vsrai_h, simd_shr, m128i, i16x8);
impl_vuv!("lsx", lsx_vsrai_w, simd_shr, m128i, i32x4);
impl_vuv!("lsx", lsx_vsrai_d, simd_shr, m128i, i64x2);
impl_vuv!("lsx", lsx_vsrli_b, simd_shr, m128i, u8x16);
impl_vuv!("lsx", lsx_vsrli_h, simd_shr, m128i, u16x8);
impl_vuv!("lsx", lsx_vsrli_w, simd_shr, m128i, u32x4);
impl_vuv!("lsx", lsx_vsrli_d, simd_shr, m128i, u64x2);
impl_vuv!("lsx", lsx_vrotri_b, simd_ext_rotr, m128i, u8x16);
impl_vuv!("lsx", lsx_vrotri_h, simd_ext_rotr, m128i, u16x8);
impl_vuv!("lsx", lsx_vrotri_w, simd_ext_rotr, m128i, u32x4);
impl_vuv!("lsx", lsx_vrotri_d, simd_ext_rotr, m128i, u64x2);
impl_vuv!("lsx", lsx_vaddi_bu, simd_add, m128i, u8x16, 5);
impl_vuv!("lsx", lsx_vaddi_hu, simd_add, m128i, u16x8, 5);
impl_vuv!("lsx", lsx_vaddi_wu, simd_add, m128i, u32x4, 5);
impl_vuv!("lsx", lsx_vaddi_du, simd_add, m128i, u64x2, 5);
impl_vuv!("lsx", lsx_vslti_bu, simd_lt, m128i, u8x16, 5);
impl_vuv!("lsx", lsx_vslti_hu, simd_lt, m128i, u16x8, 5);
impl_vuv!("lsx", lsx_vslti_wu, simd_lt, m128i, u32x4, 5);
impl_vuv!("lsx", lsx_vslti_du, simd_lt, m128i, u64x2, 5);
impl_vuv!("lsx", lsx_vslei_bu, simd_le, m128i, u8x16, 5);
impl_vuv!("lsx", lsx_vslei_hu, simd_le, m128i, u16x8, 5);
impl_vuv!("lsx", lsx_vslei_wu, simd_le, m128i, u32x4, 5);
impl_vuv!("lsx", lsx_vslei_du, simd_le, m128i, u64x2, 5);
impl_vuv!("lsx", lsx_vmaxi_bu, simd_imax, m128i, u8x16, 5);
impl_vuv!("lsx", lsx_vmaxi_hu, simd_imax, m128i, u16x8, 5);
impl_vuv!("lsx", lsx_vmaxi_wu, simd_imax, m128i, u32x4, 5);
impl_vuv!("lsx", lsx_vmaxi_du, simd_imax, m128i, u64x2, 5);
impl_vuv!("lsx", lsx_vmini_bu, simd_imin, m128i, u8x16, 5);
impl_vuv!("lsx", lsx_vmini_hu, simd_imin, m128i, u16x8, 5);
impl_vuv!("lsx", lsx_vmini_wu, simd_imin, m128i, u32x4, 5);
impl_vuv!("lsx", lsx_vmini_du, simd_imin, m128i, u64x2, 5);
impl_vuv!("lsx", lsx_vreplvei_b, simd_ext_replvei_b, m128i, i8x16, 4, const);
impl_vuv!("lsx", lsx_vreplvei_h, simd_ext_replvei_h, m128i, i16x8, 3, const);
impl_vuv!("lsx", lsx_vreplvei_w, simd_ext_replvei_w, m128i, i32x4, 2, const);
impl_vuv!("lsx", lsx_vreplvei_d, simd_ext_replvei_d, m128i, i64x2, 1, const);
impl_vuv!("lsx", lsx_vshuf4i_b, simd_ext_shuf4i_b, m128i, i8x16, 8, const);
impl_vuv!("lsx", lsx_vshuf4i_h, simd_ext_shuf4i_h, m128i, i16x8, 8, const);
impl_vuv!("lsx", lsx_vshuf4i_w, simd_ext_shuf4i_w, m128i, i32x4, 8, const);
impl_vuv!("lsx", lsx_vbsll_v, simd_ext_bsll, m128i, i8x16, 5, const);
impl_vuv!("lsx", lsx_vbsrl_v, simd_ext_bsrl, m128i, i8x16, 5, const);

impl_vug!("lsx", lsx_vpickve2gr_b, simd_extract, m128i, i8x16, i32, 4);
impl_vug!("lsx", lsx_vpickve2gr_h, simd_extract, m128i, i16x8, i32, 3);
impl_vug!("lsx", lsx_vpickve2gr_w, simd_extract, m128i, i32x4, i32, 2);
impl_vug!("lsx", lsx_vpickve2gr_d, simd_extract, m128i, i64x2, i64, 1);
impl_vug!("lsx", lsx_vpickve2gr_bu, simd_extract, m128i, u8x16, u32, 4);
impl_vug!("lsx", lsx_vpickve2gr_hu, simd_extract, m128i, u16x8, u32, 3);
impl_vug!("lsx", lsx_vpickve2gr_wu, simd_extract, m128i, u32x4, u32, 2);
impl_vug!("lsx", lsx_vpickve2gr_du, simd_extract, m128i, u64x2, u64, 1);

impl_vsv!("lsx", lsx_vseqi_b, simd_eq, m128i, i8x16, 5);
impl_vsv!("lsx", lsx_vseqi_h, simd_eq, m128i, i16x8, 5);
impl_vsv!("lsx", lsx_vseqi_w, simd_eq, m128i, i32x4, 5);
impl_vsv!("lsx", lsx_vseqi_d, simd_eq, m128i, i64x2, 5);
impl_vsv!("lsx", lsx_vslti_b, simd_lt, m128i, i8x16, 5);
impl_vsv!("lsx", lsx_vslti_h, simd_lt, m128i, i16x8, 5);
impl_vsv!("lsx", lsx_vslti_w, simd_lt, m128i, i32x4, 5);
impl_vsv!("lsx", lsx_vslti_d, simd_lt, m128i, i64x2, 5);
impl_vsv!("lsx", lsx_vslei_b, simd_le, m128i, i8x16, 5);
impl_vsv!("lsx", lsx_vslei_h, simd_le, m128i, i16x8, 5);
impl_vsv!("lsx", lsx_vslei_w, simd_le, m128i, i32x4, 5);
impl_vsv!("lsx", lsx_vslei_d, simd_le, m128i, i64x2, 5);
impl_vsv!("lsx", lsx_vmaxi_b, simd_imax, m128i, i8x16, 5);
impl_vsv!("lsx", lsx_vmaxi_h, simd_imax, m128i, i16x8, 5);
impl_vsv!("lsx", lsx_vmaxi_w, simd_imax, m128i, i32x4, 5);
impl_vsv!("lsx", lsx_vmaxi_d, simd_imax, m128i, i64x2, 5);
impl_vsv!("lsx", lsx_vmini_b, simd_imin, m128i, i8x16, 5);
impl_vsv!("lsx", lsx_vmini_h, simd_imin, m128i, i16x8, 5);
impl_vsv!("lsx", lsx_vmini_w, simd_imin, m128i, i32x4, 5);
impl_vsv!("lsx", lsx_vmini_d, simd_imin, m128i, i64x2, 5);

impl_vvvv!("lsx", lsx_vmadd_b, simd_ext_madd, m128i, i8x16);
impl_vvvv!("lsx", lsx_vmadd_h, simd_ext_madd, m128i, i16x8);
impl_vvvv!("lsx", lsx_vmadd_w, simd_ext_madd, m128i, i32x4);
impl_vvvv!("lsx", lsx_vmadd_d, simd_ext_madd, m128i, i64x2);
impl_vvvv!("lsx", lsx_vmsub_b, simd_ext_msub, m128i, i8x16);
impl_vvvv!("lsx", lsx_vmsub_h, simd_ext_msub, m128i, i16x8);
impl_vvvv!("lsx", lsx_vmsub_w, simd_ext_msub, m128i, i32x4);
impl_vvvv!("lsx", lsx_vmsub_d, simd_ext_msub, m128i, i64x2);
impl_vvvv!("lsx", lsx_vfmadd_s, simd_fma, m128, f32x4);
impl_vvvv!("lsx", lsx_vfmadd_d, simd_fma, m128d, f64x2);
impl_vvvv!("lsx", lsx_vfmsub_s, simd_ext_fmsub, m128, f32x4);
impl_vvvv!("lsx", lsx_vfmsub_d, simd_ext_fmsub, m128d, f64x2);
impl_vvvv!("lsx", lsx_vfnmadd_s, simd_ext_fnmadd, m128, f32x4);
impl_vvvv!("lsx", lsx_vfnmadd_d, simd_ext_fnmadd, m128d, f64x2);
impl_vvvv!("lsx", lsx_vfnmsub_s, simd_ext_fnmsub, m128, f32x4);
impl_vvvv!("lsx", lsx_vfnmsub_d, simd_ext_fnmsub, m128d, f64x2);

impl_vvuv!("lsx", lsx_vshuf4i_d, simd_ext_shuf4i_d, m128i, i64x2, 8, const);

impl_vugv!("lsx", lsx_vinsgr2vr_b, simd_insert, m128i, i8x16, i32, 4);
impl_vugv!("lsx", lsx_vinsgr2vr_h, simd_insert, m128i, i16x8, i32, 3);
impl_vugv!("lsx", lsx_vinsgr2vr_w, simd_insert, m128i, i32x4, i32, 2);
impl_vugv!("lsx", lsx_vinsgr2vr_d, simd_insert, m128i, i64x2, i64, 1);

#[cfg(test)]
mod tests {
    use crate::{
        core_arch::{loongarch64::*, simd::*},
        mem::transmute,
    };
    use std::hint::black_box;
    use stdarch_test::simd_test;

    #[simd_test(enable = "lsx")]
    unsafe fn vldi() {
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<0>()));
        let r = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        assert_eq!(r, transmute(lsx_vldi::<255>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<1024>()));
        let r = i8x16::new(0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2);
        assert_eq!(r, transmute(lsx_vldi::<1536>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<2048>()));
        let r = i8x16::new(0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1, -1, 0, -2, -1, -1);
        assert_eq!(r, transmute(lsx_vldi::<2560>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<3072>()));
        let r = i8x16::new(0, -2, -1, -1, -1, -1, -1, -1, 0, -2, -1, -1, -1, -1, -1, -1);
        assert_eq!(r, transmute(lsx_vldi::<3584>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-4096>()));
        let r = i8x16::new(-128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3968>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3840>()));
        let r = i8x16::new(0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3712>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3584>()));
        let r = i8x16::new(0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3456>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3328>()));
        let r = i8x16::new(0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128, 0, 0, 0, -128);
        assert_eq!(r, transmute(lsx_vldi::<-3200>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-3072>()));
        let r = i8x16::new(
            -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0,
        );
        assert_eq!(r, transmute(lsx_vldi::<-2944>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-2816>()));
        let r = i8x16::new(
            0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128, 0, -128,
        );
        assert_eq!(r, transmute(lsx_vldi::<-2688>()));
        let r = i8x16::new(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-2560>()));
        let r = i8x16::new(
            -1, -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0, -1, -128, 0, 0,
        );
        assert_eq!(r, transmute(lsx_vldi::<-2432>()));
        let r = i8x16::new(-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-2304>()));
        let r = i8x16::new(
            -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0, -1, -1, -128, 0,
        );
        assert_eq!(r, transmute(lsx_vldi::<-2176>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-2048>()));
        let r = i8x16::new(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        assert_eq!(r, transmute(lsx_vldi::<-1793>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-1792>()));
        let r = i8x16::new(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
        assert_eq!(r, transmute(lsx_vldi::<-1622>()));
        let r = i8x16::new(0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64);
        assert_eq!(r, transmute(lsx_vldi::<-1536>()));
        let r = i8x16::new(0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80, -63, 0, 0, 80, -63);
        assert_eq!(r, transmute(lsx_vldi::<-1366>()));
        let r = i8x16::new(0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-1280>()));
        let r = i8x16::new(0, 0, 80, -63, 0, 0, 0, 0, 0, 0, 80, -63, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vldi::<-1110>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64);
        assert_eq!(r, transmute(lsx_vldi::<-1024>()));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 42, -64, 0, 0, 0, 0, 0, 0, 42, -64);
        assert_eq!(r, transmute(lsx_vldi::<-854>()));
    }

    #[simd_test(enable = "lsx")]
    unsafe fn vbsll_v() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(r, transmute(lsx_vbsll_v::<0>(black_box(transmute(a)))));
        let r = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(r, transmute(lsx_vbsll_v::<1>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
        assert_eq!(r, transmute(lsx_vbsll_v::<2>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
        assert_eq!(r, transmute(lsx_vbsll_v::<3>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        assert_eq!(r, transmute(lsx_vbsll_v::<4>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        assert_eq!(r, transmute(lsx_vbsll_v::<5>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        assert_eq!(r, transmute(lsx_vbsll_v::<6>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(r, transmute(lsx_vbsll_v::<7>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(r, transmute(lsx_vbsll_v::<8>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(r, transmute(lsx_vbsll_v::<9>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6);
        assert_eq!(r, transmute(lsx_vbsll_v::<10>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5);
        assert_eq!(r, transmute(lsx_vbsll_v::<11>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4);
        assert_eq!(r, transmute(lsx_vbsll_v::<12>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3);
        assert_eq!(r, transmute(lsx_vbsll_v::<13>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2);
        assert_eq!(r, transmute(lsx_vbsll_v::<14>(black_box(transmute(a)))));
        let r = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        assert_eq!(r, transmute(lsx_vbsll_v::<15>(black_box(transmute(a)))));
    }

    #[simd_test(enable = "lsx")]
    unsafe fn vbsrl_v() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(r, transmute(lsx_vbsrl_v::<0>(black_box(transmute(a)))));
        let r = i8x16::new(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<1>(black_box(transmute(a)))));
        let r = i8x16::new(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<2>(black_box(transmute(a)))));
        let r = i8x16::new(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<3>(black_box(transmute(a)))));
        let r = i8x16::new(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<4>(black_box(transmute(a)))));
        let r = i8x16::new(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<5>(black_box(transmute(a)))));
        let r = i8x16::new(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<6>(black_box(transmute(a)))));
        let r = i8x16::new(8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<7>(black_box(transmute(a)))));
        let r = i8x16::new(9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<8>(black_box(transmute(a)))));
        let r = i8x16::new(10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<9>(black_box(transmute(a)))));
        let r = i8x16::new(11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<10>(black_box(transmute(a)))));
        let r = i8x16::new(12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<11>(black_box(transmute(a)))));
        let r = i8x16::new(13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<12>(black_box(transmute(a)))));
        let r = i8x16::new(14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<13>(black_box(transmute(a)))));
        let r = i8x16::new(15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<14>(black_box(transmute(a)))));
        let r = i8x16::new(16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, transmute(lsx_vbsrl_v::<15>(black_box(transmute(a)))));
    }
}
