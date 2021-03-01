//! Tests for ARM+v7+neon shift and insert (vsli[q]_n, vsri[q]_n) intrinsics.
//!
//! These are included in `{arm, aarch64}::neon`.

use super::*;

#[cfg(target_arch = "aarch64")]
use crate::core_arch::aarch64::*;

#[cfg(target_arch = "arm")]
use crate::core_arch::arm::*;

use crate::core_arch::simd::*;
use std::mem::transmute;
use stdarch_test::simd_test;

macro_rules! test_vsli {
    ($test_id:ident, $t:ty => $fn_id:ident ([$($a:expr),*], [$($b:expr),*], $n:expr)) => {
        #[simd_test(enable = "neon")]
        #[allow(unused_assignments)]
        unsafe fn $test_id() {
            let a = [$($a as $t),*];
            let b = [$($b as $t),*];
            let n_bit_mask: $t = (1 << $n) - 1;
            let e = [$(($a as $t & n_bit_mask) | ($b as $t << $n)),*];
            let r = $fn_id::<$n>(transmute(a), transmute(b));
            let mut d = e;
            d = transmute(r);
            assert_eq!(d, e);
        }
    }
}
test_vsli!(test_vsli_n_s8, i8 => vsli_n_s8([3, -44, 127, -56, 0, 24, -97, 10], [-128, -14, 125, -77, 27, 8, -1, 110], 5));
test_vsli!(test_vsliq_n_s8, i8 => vsliq_n_s8([3, -44, 127, -56, 0, 24, -97, 10, -33, 1, -6, -39, 15, 101, -80, -1], [-128, -14, 125, -77, 27, 8, -1, 110, -4, -92, 111, 32, 1, -4, -29, 99], 2));
test_vsli!(test_vsli_n_s16, i16 => vsli_n_s16([3304, -44, 2300, -546], [-1208, -140, 1225, -707], 7));
test_vsli!(test_vsliq_n_s16, i16 => vsliq_n_s16([3304, -44, 2300, -20046, 0, 9924, -907, 1190], [-1208, -140, 4225, -707, 2701, 804, -71, 2110], 14));
test_vsli!(test_vsli_n_s32, i32 => vsli_n_s32([125683, -78901], [-128, -112944], 23));
test_vsli!(test_vsliq_n_s32, i32 => vsliq_n_s32([125683, -78901, 127, -12009], [-128, -112944, 125, -707], 15));
test_vsli!(test_vsli_n_s64, i64 => vsli_n_s64([-333333], [1028], 45));
test_vsli!(test_vsliq_n_s64, i64 => vsliq_n_s64([-333333, -52023], [1028, -99814], 33));
test_vsli!(test_vsli_n_u8, u8 => vsli_n_u8([3, 44, 127, 56, 0, 24, 97, 10], [127, 14, 125, 77, 27, 8, 1, 110], 5));
test_vsli!(test_vsliq_n_u8, u8 => vsliq_n_u8([3, 44, 127, 56, 0, 24, 97, 10, 33, 1, 6, 39, 15, 101, 80, 1], [127, 14, 125, 77, 27, 8, 1, 110, 4, 92, 111, 32, 1, 4, 29, 99], 2));
test_vsli!(test_vsli_n_u16, u16 => vsli_n_u16([3304, 44, 2300, 546], [1208, 140, 1225, 707], 7));
test_vsli!(test_vsliq_n_u16, u16 => vsliq_n_u16([3304, 44, 2300, 20046, 0, 9924, 907, 1190], [1208, 140, 4225, 707, 2701, 804, 71, 2110], 14));
test_vsli!(test_vsli_n_u32, u32 => vsli_n_u32([125683, 78901], [128, 112944], 23));
test_vsli!(test_vsliq_n_u32, u32 => vsliq_n_u32([125683, 78901, 127, 12009], [128, 112944, 125, 707], 15));
test_vsli!(test_vsli_n_u64, u64 => vsli_n_u64([333333], [1028], 45));
test_vsli!(test_vsliq_n_u64, u64 => vsliq_n_u64([333333, 52023], [1028, 99814], 33));
test_vsli!(test_vsli_n_p8, i8 => vsli_n_p8([3, 44, 127, 56, 0, 24, 97, 10], [127, 14, 125, 77, 27, 8, 1, 110], 5));
test_vsli!(test_vsliq_n_p8, i8 => vsliq_n_p8([3, 44, 127, 56, 0, 24, 97, 10, 33, 1, 6, 39, 15, 101, 80, 1], [127, 14, 125, 77, 27, 8, 1, 110, 4, 92, 111, 32, 1, 4, 29, 99], 2));
test_vsli!(test_vsli_n_p16, i16 => vsli_n_p16([3304, 44, 2300, 546], [1208, 140, 1225, 707], 7));
test_vsli!(test_vsliq_n_p16, i16 => vsliq_n_p16([3304, 44, 2300, 20046, 0, 9924, 907, 1190], [1208, 140, 4225, 707, 2701, 804, 71, 2110], 14));

macro_rules! test_vsri {
    ($test_id:ident, $t:ty => $fn_id:ident ([$($a:expr),*], [$($b:expr),*], $n:expr)) => {
        #[simd_test(enable = "neon")]
        #[allow(unused_assignments)]
        unsafe fn $test_id() {
            let a = [$($a as $t),*];
            let b = [$($b as $t),*];
            let n_bit_mask = ((1 as $t << $n) - 1).rotate_right($n);
            let e = [$(($a as $t & n_bit_mask) | (($b as $t >> $n) & !n_bit_mask)),*];
            let r = $fn_id(transmute(a), transmute(b), $n);
            let mut d = e;
            d = transmute(r);
            assert_eq!(d, e);
        }
    }
}
test_vsri!(test_vsri_n_s8, i8 => vsri_n_s8([3, -44, 127, -56, 0, 24, -97, 10], [-128, -14, 125, -77, 27, 8, -1, 110], 5));
test_vsri!(test_vsriq_n_s8, i8 => vsriq_n_s8([3, -44, 127, -56, 0, 24, -97, 10, -33, 1, -6, -39, 15, 101, -80, -1], [-128, -14, 125, -77, 27, 8, -1, 110, -4, -92, 111, 32, 1, -4, -29, 99], 2));
test_vsri!(test_vsri_n_s16, i16 => vsri_n_s16([3304, -44, 2300, -546], [-1208, -140, 1225, -707], 7));
test_vsri!(test_vsriq_n_s16, i16 => vsriq_n_s16([3304, -44, 2300, -20046, 0, 9924, -907, 1190], [-1208, -140, 4225, -707, 2701, 804, -71, 2110], 14));
test_vsri!(test_vsri_n_s32, i32 => vsri_n_s32([125683, -78901], [-128, -112944], 23));
test_vsri!(test_vsriq_n_s32, i32 => vsriq_n_s32([125683, -78901, 127, -12009], [-128, -112944, 125, -707], 15));
test_vsri!(test_vsri_n_s64, i64 => vsri_n_s64([-333333], [1028], 45));
test_vsri!(test_vsriq_n_s64, i64 => vsriq_n_s64([-333333, -52023], [1028, -99814], 33));
test_vsri!(test_vsri_n_u8, u8 => vsri_n_u8([3, 44, 127, 56, 0, 24, 97, 10], [127, 14, 125, 77, 27, 8, 1, 110], 5));
test_vsri!(test_vsriq_n_u8, u8 => vsriq_n_u8([3, 44, 127, 56, 0, 24, 97, 10, 33, 1, 6, 39, 15, 101, 80, 1], [127, 14, 125, 77, 27, 8, 1, 110, 4, 92, 111, 32, 1, 4, 29, 99], 2));
test_vsri!(test_vsri_n_u16, u16 => vsri_n_u16([3304, 44, 2300, 546], [1208, 140, 1225, 707], 7));
test_vsri!(test_vsriq_n_u16, u16 => vsriq_n_u16([3304, 44, 2300, 20046, 0, 9924, 907, 1190], [1208, 140, 4225, 707, 2701, 804, 71, 2110], 14));
test_vsri!(test_vsri_n_u32, u32 => vsri_n_u32([125683, 78901], [128, 112944], 23));
test_vsri!(test_vsriq_n_u32, u32 => vsriq_n_u32([125683, 78901, 127, 12009], [128, 112944, 125, 707], 15));
test_vsri!(test_vsri_n_u64, u64 => vsri_n_u64([333333], [1028], 45));
test_vsri!(test_vsriq_n_u64, u64 => vsriq_n_u64([333333, 52023], [1028, 99814], 33));
test_vsri!(test_vsri_n_p8, i8 => vsri_n_p8([3, 44, 127, 56, 0, 24, 97, 10], [127, 14, 125, 77, 27, 8, 1, 110], 5));
test_vsri!(test_vsriq_n_p8, i8 => vsriq_n_p8([3, 44, 127, 56, 0, 24, 97, 10, 33, 1, 6, 39, 15, 101, 80, 1], [127, 14, 125, 77, 27, 8, 1, 110, 4, 92, 111, 32, 1, 4, 29, 99], 2));
test_vsri!(test_vsri_n_p16, i16 => vsri_n_p16([3304, 44, 2300, 546], [1208, 140, 1225, 707], 7));
test_vsri!(test_vsriq_n_p16, i16 => vsriq_n_p16([3304, 44, 2300, 20046, 0, 9924, 907, 1190], [1208, 140, 4225, 707, 2701, 804, 71, 2110], 14));
