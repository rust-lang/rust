//! This module implements the [WebAssembly `SIMD128` ISA].
//!
//! [WebAssembly `SIMD128` ISA]:
//! https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md

#![unstable(feature = "wasm_simd", issue = "74372")]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

use crate::{
    core_arch::{simd::*, simd_llvm::*},
    marker::Sized,
    mem::transmute,
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    /// WASM-specific 128-bit wide SIMD vector type.
    // N.B., internals here are arbitrary.
    pub struct v128(i32, i32, i32, i32);
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait v128Ext: Sized {
    unsafe fn as_v128(self) -> v128;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_u8x16(self) -> u8x16 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_u16x8(self) -> u16x8 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_u32x4(self) -> u32x4 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_u64x2(self) -> u64x2 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_i8x16(self) -> i8x16 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_i16x8(self) -> i16x8 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_i32x4(self) -> i32x4 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_i64x2(self) -> i64x2 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_f32x4(self) -> f32x4 {
        transmute(self.as_v128())
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_f64x2(self) -> f64x2 {
        transmute(self.as_v128())
    }
}

impl v128Ext for v128 {
    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn as_v128(self) -> Self {
        self
    }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.wasm.load32.zero"]
    fn llvm_load32_zero(x: *const u32) -> i32x4;
    #[link_name = "llvm.wasm.load64.zero"]
    fn llvm_load64_zero(x: *const u64) -> i64x2;
    #[link_name = "llvm.wasm.load8.lane"]
    fn llvm_load8_lane(x: *const u8, v: u8x16, l: usize) -> u8x16;
    #[link_name = "llvm.wasm.load16.lane"]
    fn llvm_load16_lane(x: *const u16, v: u16x8, l: usize) -> u16x8;
    #[link_name = "llvm.wasm.load32.lane"]
    fn llvm_load32_lane(x: *const u32, v: u32x4, l: usize) -> u32x4;
    #[link_name = "llvm.wasm.load64.lane"]
    fn llvm_load64_lane(x: *const u64, v: u64x2, l: usize) -> u64x2;
    #[link_name = "llvm.wasm.store8.lane"]
    fn llvm_store8_lane(x: *mut u8, v: u8x16, l: usize);
    #[link_name = "llvm.wasm.store16.lane"]
    fn llvm_store16_lane(x: *mut u16, v: u16x8, l: usize);
    #[link_name = "llvm.wasm.store32.lane"]
    fn llvm_store32_lane(x: *mut u32, v: u32x4, l: usize);
    #[link_name = "llvm.wasm.store64.lane"]
    fn llvm_store64_lane(x: *mut u64, v: u64x2, l: usize);

    #[link_name = "llvm.wasm.swizzle"]
    fn llvm_swizzle(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.eq"]
    fn llvm_eq(a: i64x2, b: i64x2) -> i64x2;

    #[link_name = "llvm.wasm.bitselect.v16i8"]
    fn llvm_bitselect(a: i8x16, b: i8x16, c: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.anytrue.v16i8"]
    fn llvm_any_true_i8x16(x: i8x16) -> i32;

    #[link_name = "llvm.wasm.alltrue.v16i8"]
    fn llvm_i8x16_all_true(x: i8x16) -> i32;
    #[link_name = "llvm.wasm.popcnt"]
    fn llvm_popcnt(a: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.bitmask.v16i8"]
    fn llvm_bitmask_i8x16(a: i8x16) -> i32;
    #[link_name = "llvm.wasm.narrow.signed.v16i8.v8i16"]
    fn llvm_narrow_i8x16_s(a: i16x8, b: i16x8) -> i8x16;
    #[link_name = "llvm.wasm.narrow.unsigned.v16i8.v8i16"]
    fn llvm_narrow_i8x16_u(a: i16x8, b: i16x8) -> i8x16;
    #[link_name = "llvm.sadd.sat.v16i8"]
    fn llvm_i8x16_add_sat_s(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.uadd.sat.v16i8"]
    fn llvm_i8x16_add_sat_u(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.sub.saturate.signed.v16i8"]
    fn llvm_i8x16_sub_sat_s(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.sub.saturate.unsigned.v16i8"]
    fn llvm_i8x16_sub_sat_u(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.avgr.unsigned.v16i8"]
    fn llvm_avgr_u_i8x16(a: i8x16, b: i8x16) -> i8x16;

    #[link_name = "llvm.wasm.extadd.pairwise.signed.v8i16"]
    fn llvm_i16x8_extadd_pairwise_i8x16_s(x: i8x16) -> i16x8;
    #[link_name = "llvm.wasm.extadd.pairwise.unsigned.v8i16"]
    fn llvm_i16x8_extadd_pairwise_i8x16_u(x: i8x16) -> i16x8;
    #[link_name = "llvm.wasm.q15mulr.saturate.signed"]
    fn llvm_q15mulr(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.alltrue.v8i16"]
    fn llvm_i16x8_all_true(x: i16x8) -> i32;
    #[link_name = "llvm.wasm.bitmask.v8i16"]
    fn llvm_bitmask_i16x8(a: i16x8) -> i32;
    #[link_name = "llvm.wasm.narrow.signed.v8i16.v8i16"]
    fn llvm_narrow_i16x8_s(a: i32x4, b: i32x4) -> i16x8;
    #[link_name = "llvm.wasm.narrow.unsigned.v8i16.v8i16"]
    fn llvm_narrow_i16x8_u(a: i32x4, b: i32x4) -> i16x8;
    #[link_name = "llvm.sadd.sat.v8i16"]
    fn llvm_i16x8_add_sat_s(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.uadd.sat.v8i16"]
    fn llvm_i16x8_add_sat_u(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.sub.saturate.signed.v8i16"]
    fn llvm_i16x8_sub_sat_s(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.sub.saturate.unsigned.v8i16"]
    fn llvm_i16x8_sub_sat_u(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.avgr.unsigned.v8i16"]
    fn llvm_avgr_u_i16x8(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.extmul.low.signed.v8i16"]
    fn llvm_i16x8_extmul_low_i8x16_s(a: i8x16, b: i8x16) -> i16x8;
    #[link_name = "llvm.wasm.extmul.high.signed.v8i16"]
    fn llvm_i16x8_extmul_high_i8x16_s(a: i8x16, b: i8x16) -> i16x8;
    #[link_name = "llvm.wasm.extmul.low.unsigned.v8i16"]
    fn llvm_i16x8_extmul_low_i8x16_u(a: i8x16, b: i8x16) -> i16x8;
    #[link_name = "llvm.wasm.extmul.high.unsigned.v8i16"]
    fn llvm_i16x8_extmul_high_i8x16_u(a: i8x16, b: i8x16) -> i16x8;

    #[link_name = "llvm.wasm.extadd.pairwise.signed.v16i8"]
    fn llvm_i32x4_extadd_pairwise_i16x8_s(x: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.extadd.pairwise.unsigned.v16i8"]
    fn llvm_i32x4_extadd_pairwise_i16x8_u(x: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.alltrue.v4i32"]
    fn llvm_i32x4_all_true(x: i32x4) -> i32;
    #[link_name = "llvm.wasm.bitmask.v4i32"]
    fn llvm_bitmask_i32x4(a: i32x4) -> i32;
    #[link_name = "llvm.wasm.dot"]
    fn llvm_i32x4_dot_i16x8_s(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.extmul.low.signed.v4i32"]
    fn llvm_i32x4_extmul_low_i16x8_s(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.extmul.high.signed.v4i32"]
    fn llvm_i32x4_extmul_high_i16x8_s(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.extmul.low.unsigned.v4i32"]
    fn llvm_i32x4_extmul_low_i16x8_u(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.wasm.extmul.high.unsigned.v4i32"]
    fn llvm_i32x4_extmul_high_i16x8_u(a: i16x8, b: i16x8) -> i32x4;

    #[link_name = "llvm.wasm.alltrue.v2i64"]
    fn llvm_i64x2_all_true(x: i64x2) -> i32;
    #[link_name = "llvm.wasm.bitmask.v2i64"]
    fn llvm_bitmask_i64x2(a: i64x2) -> i32;
    #[link_name = "llvm.wasm.extmul.low.signed.v2i64"]
    fn llvm_i64x2_extmul_low_i32x4_s(a: i32x4, b: i32x4) -> i64x2;
    #[link_name = "llvm.wasm.extmul.high.signed.v2i64"]
    fn llvm_i64x2_extmul_high_i32x4_s(a: i32x4, b: i32x4) -> i64x2;
    #[link_name = "llvm.wasm.extmul.low.unsigned.v2i64"]
    fn llvm_i64x2_extmul_low_i32x4_u(a: i32x4, b: i32x4) -> i64x2;
    #[link_name = "llvm.wasm.extmul.high.unsigned.v2i64"]
    fn llvm_i64x2_extmul_high_i32x4_u(a: i32x4, b: i32x4) -> i64x2;

    #[link_name = "llvm.wasm.ceil.v4f32"]
    fn llvm_f32x4_ceil(x: f32x4) -> f32x4;
    #[link_name = "llvm.wasm.floor.v4f32"]
    fn llvm_f32x4_floor(x: f32x4) -> f32x4;
    #[link_name = "llvm.wasm.trunc.v4f32"]
    fn llvm_f32x4_trunc(x: f32x4) -> f32x4;
    #[link_name = "llvm.wasm.nearest.v4f32"]
    fn llvm_f32x4_nearest(x: f32x4) -> f32x4;
    #[link_name = "llvm.fabs.v4f32"]
    fn llvm_f32x4_abs(x: f32x4) -> f32x4;
    #[link_name = "llvm.sqrt.v4f32"]
    fn llvm_f32x4_sqrt(x: f32x4) -> f32x4;
    #[link_name = "llvm.minimum.v4f32"]
    fn llvm_f32x4_min(x: f32x4, y: f32x4) -> f32x4;
    #[link_name = "llvm.maximum.v4f32"]
    fn llvm_f32x4_max(x: f32x4, y: f32x4) -> f32x4;
    #[link_name = "llvm.wasm.pmin.v4f32"]
    fn llvm_f32x4_pmin(x: f32x4, y: f32x4) -> f32x4;
    #[link_name = "llvm.wasm.pmax.v4f32"]
    fn llvm_f32x4_pmax(x: f32x4, y: f32x4) -> f32x4;

    #[link_name = "llvm.wasm.ceil.v2f64"]
    fn llvm_f64x2_ceil(x: f64x2) -> f64x2;
    #[link_name = "llvm.wasm.floor.v2f64"]
    fn llvm_f64x2_floor(x: f64x2) -> f64x2;
    #[link_name = "llvm.wasm.trunc.v2f64"]
    fn llvm_f64x2_trunc(x: f64x2) -> f64x2;
    #[link_name = "llvm.wasm.nearest.v2f64"]
    fn llvm_f64x2_nearest(x: f64x2) -> f64x2;
    #[link_name = "llvm.fabs.v2f64"]
    fn llvm_f64x2_abs(x: f64x2) -> f64x2;
    #[link_name = "llvm.sqrt.v2f64"]
    fn llvm_f64x2_sqrt(x: f64x2) -> f64x2;
    #[link_name = "llvm.minimum.v2f64"]
    fn llvm_f64x2_min(x: f64x2, y: f64x2) -> f64x2;
    #[link_name = "llvm.maximum.v2f64"]
    fn llvm_f64x2_max(x: f64x2, y: f64x2) -> f64x2;
    #[link_name = "llvm.wasm.pmin.v2f64"]
    fn llvm_f64x2_pmin(x: f64x2, y: f64x2) -> f64x2;
    #[link_name = "llvm.wasm.pmax.v2f64"]
    fn llvm_f64x2_pmax(x: f64x2, y: f64x2) -> f64x2;

    #[link_name = "llvm.wasm.convert.low.signed"]
    fn llvm_f64x2_convert_low_i32x4_s(x: i32x4) -> f64x2;
    #[link_name = "llvm.wasm.convert.low.unsigned"]
    fn llvm_f64x2_convert_low_i32x4_u(x: i32x4) -> f64x2;
    #[link_name = "llvm.wasm.trunc.saturate.zero.signed"]
    fn llvm_i32x4_trunc_sat_f64x2_s_zero(x: f64x2) -> i32x4;
    #[link_name = "llvm.wasm.trunc.saturate.zero.unsigned"]
    fn llvm_i32x4_trunc_sat_f64x2_u_zero(x: f64x2) -> i32x4;
    #[link_name = "llvm.wasm.demote.zero"]
    fn llvm_f32x4_demote_f64x2_zero(x: f64x2) -> f32x4;
    #[link_name = "llvm.wasm.promote.low"]
    fn llvm_f64x2_promote_low_f32x4(x: f32x4) -> f64x2;
}

/// Loads a `v128` vector from the given heap address.
#[inline]
#[cfg_attr(test, assert_instr(v128.load))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    *m
}

/// Load eight 8-bit integers and sign extend each one to a 16-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load8x8_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load8x8_s(m: *const i8) -> v128 {
    transmute(simd_cast::<_, i16x8>(*(m as *const i8x8)))
}

/// Load eight 8-bit integers and zero extend each one to a 16-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load8x8_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load8x8_u(m: *const u8) -> v128 {
    transmute(simd_cast::<_, u16x8>(*(m as *const u8x8)))
}

/// Load four 16-bit integers and sign extend each one to a 32-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load16x4_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load16x4_s(m: *const i16) -> v128 {
    transmute(simd_cast::<_, i32x4>(*(m as *const i16x4)))
}

/// Load four 16-bit integers and zero extend each one to a 32-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load16x4_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load16x4_u(m: *const u16) -> v128 {
    transmute(simd_cast::<_, u32x4>(*(m as *const u16x4)))
}

/// Load two 32-bit integers and sign extend each one to a 64-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load32x2_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load32x2_s(m: *const i32) -> v128 {
    transmute(simd_cast::<_, i64x2>(*(m as *const i32x2)))
}

/// Load two 32-bit integers and zero extend each one to a 64-bit lane
#[inline]
#[cfg_attr(test, assert_instr(v128.load32x2_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load32x2_u(m: *const u32) -> v128 {
    transmute(simd_cast::<_, u64x2>(*(m as *const u32x2)))
}

/// Load a single element and splat to all lanes of a v128 vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.load8_splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load8_splat(m: *const u8) -> v128 {
    transmute(u8x16::splat(*m))
}

/// Load a single element and splat to all lanes of a v128 vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.load16_splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load16_splat(m: *const u16) -> v128 {
    transmute(u16x8::splat(*m))
}

/// Load a single element and splat to all lanes of a v128 vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32_splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load32_splat(m: *const u32) -> v128 {
    transmute(u32x4::splat(*m))
}

/// Load a single element and splat to all lanes of a v128 vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.load64_splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load64_splat(m: *const u64) -> v128 {
    transmute(u64x2::splat(*m))
}

/// Load a 32-bit element into the low bits of the vector and sets all other
/// bits to zero.
#[inline]
// #[cfg_attr(test, assert_instr(v128.load32_zero))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load32_zero(m: *const u32) -> v128 {
    transmute(llvm_load32_zero(m))
}

/// Load a 64-bit element into the low bits of the vector and sets all other
/// bits to zero.
#[inline]
// #[cfg_attr(test, assert_instr(v128.load64_zero))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load64_zero(m: *const u64) -> v128 {
    transmute(llvm_load64_zero(m))
}

/// Stores a `v128` vector to the given heap address.
#[inline]
#[cfg_attr(test, assert_instr(v128.store))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    *m = a;
}

/// Loads an 8-bit value from `m` and sets lane `L` of `v` to that value.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load8_lane<const L: usize>(v: v128, m: *const u8) -> v128 {
    transmute(llvm_load8_lane(m, v.as_u8x16(), L))
}

// #[cfg(test)]
// #[assert_instr(v128.load8_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_load8_lane_test(v: v128, m: *const u8) -> v128 {
//     v128_load8_lane::<0>(v, m)
// }

/// Loads a 16-bit value from `m` and sets lane `L` of `v` to that value.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load16_lane<const L: usize>(v: v128, m: *const u16) -> v128 {
    transmute(llvm_load16_lane(m, v.as_u16x8(), L))
}

// #[cfg(test)]
// #[assert_instr(v128.load16_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_load16_lane_test(v: v128, m: *const u16) -> v128 {
//     v128_load16_lane::<0>(v, m)
// }

/// Loads a 32-bit value from `m` and sets lane `L` of `v` to that value.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load32_lane<const L: usize>(v: v128, m: *const u32) -> v128 {
    transmute(llvm_load32_lane(m, v.as_u32x4(), L))
}

// #[cfg(test)]
// #[assert_instr(v128.load32_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_load32_lane_test(v: v128, m: *const u32) -> v128 {
//     v128_load32_lane::<0>(v, m)
// }

/// Loads a 64-bit value from `m` and sets lane `L` of `v` to that value.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load64_lane<const L: usize>(v: v128, m: *const u64) -> v128 {
    transmute(llvm_load64_lane(m, v.as_u64x2(), L))
}

// #[cfg(test)]
// #[assert_instr(v128.load64_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_load64_lane_test(v: v128, m: *const u64) -> v128 {
//     v128_load64_lane::<0>(v, m)
// }

/// Stores the 8-bit value from lane `L` of `v` into `m`
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_store8_lane<const L: usize>(v: v128, m: *mut u8) {
    llvm_store8_lane(m, v.as_u8x16(), L);
}

// #[cfg(test)]
// #[assert_instr(v128.store8_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_store8_lane_test(v: v128, m: *mut u8) {
//     v128_store8_lane::<0>(v, m)
// }

/// Stores the 16-bit value from lane `L` of `v` into `m`
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_store16_lane<const L: usize>(v: v128, m: *mut u16) {
    llvm_store16_lane(m, v.as_u16x8(), L)
}

//#[cfg(test)]
//#[assert_instr(v128.store16_lane)]
//#[target_feature(enable = "simd128")]
//unsafe fn v128_store16_lane_test(v: v128, m: *mut u16) {
//    v128_store16_lane::<0>(v, m)
//}

/// Stores the 32-bit value from lane `L` of `v` into `m`
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_store32_lane<const L: usize>(v: v128, m: *mut u32) {
    llvm_store32_lane(m, v.as_u32x4(), L)
}

// #[cfg(test)]
// #[assert_instr(v128.store32_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_store32_lane_test(v: v128, m: *mut u32) {
//     v128_store32_lane::<0>(v, m)
// }

/// Stores the 64-bit value from lane `L` of `v` into `m`
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_store64_lane<const L: usize>(v: v128, m: *mut u64) {
    llvm_store64_lane(m, v.as_u64x2(), L)
}

// #[cfg(test)]
// #[assert_instr(v128.store64_lane)]
// #[target_feature(enable = "simd128")]
// unsafe fn v128_store64_lane_test(v: v128, m: *mut u64) {
//     v128_store64_lane::<0>(v, m)
// }

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as sixteen 8-bit integers.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(
//     test,
//     assert_instr(
//         v128.const,
//         a0 = 0,
//         a1 = 1,
//         a2 = 2,
//         a3 = 3,
//         a4 = 4,
//         a5 = 5,
//         a6 = 6,
//         a7 = 7,
//         a8 = 8,
//         a9 = 9,
//         a10 = 10,
//         a11 = 11,
//         a12 = 12,
//         a13 = 13,
//         a14 = 14,
//         a15 = 15,
//     )
// )]
pub const unsafe fn v128_const(
    a0: i8,
    a1: i8,
    a2: i8,
    a3: i8,
    a4: i8,
    a5: i8,
    a6: i8,
    a7: i8,
    a8: i8,
    a9: i8,
    a10: i8,
    a11: i8,
    a12: i8,
    a13: i8,
    a14: i8,
    a15: i8,
) -> v128 {
    transmute(i8x16(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
    ))
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as eight 16-bit integers.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(
//     test,
//     assert_instr(
//         v128.const,
//         a0 = 0,
//         a1 = 1,
//         a2 = 2,
//         a3 = 3,
//         a4 = 4,
//         a5 = 5,
//         a6 = 6,
//         a7 = 7,
//     )
// )]
pub const unsafe fn i16x8_const(
    a0: i16,
    a1: i16,
    a2: i16,
    a3: i16,
    a4: i16,
    a5: i16,
    a6: i16,
    a7: i16,
) -> v128 {
    transmute(i16x8(a0, a1, a2, a3, a4, a5, a6, a7))
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as four 32-bit integers.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(test, assert_instr(v128.const, a0 = 0, a1 = 1, a2 = 2, a3 = 3))]
pub const unsafe fn i32x4_const(a0: i32, a1: i32, a2: i32, a3: i32) -> v128 {
    transmute(i32x4(a0, a1, a2, a3))
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as two 64-bit integers.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(test, assert_instr(v128.const, a0 = 0, a1 = 1))]
pub const unsafe fn i64x2_const(a0: i64, a1: i64) -> v128 {
    transmute(i64x2(a0, a1))
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as four 32-bit floats.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(test, assert_instr(v128.const, a0 = 0.0, a1 = 1.0, a2 = 2.0, a3 = 3.0))]
pub const unsafe fn f32x4_const(a0: f32, a1: f32, a2: f32, a3: f32) -> v128 {
    transmute(f32x4(a0, a1, a2, a3))
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// This function generates a `v128.const` instruction as if the generated
/// vector was interpreted as two 64-bit floats.
#[inline]
#[target_feature(enable = "simd128")]
// #[cfg_attr(test, assert_instr(v128.const, a0 = 0.0, a1 = 1.0))]
pub const unsafe fn f64x2_const(a0: f64, a1: f64) -> v128 {
    transmute(f64x2(a0, a1))
}

/// Returns a new vector with lanes selected from the lanes of the two input
/// vectors `$a` and `$b` specified in the 16 immediate operands.
///
/// The `$a` and `$b` expressions must have type `v128`, and this function
/// generates a wasm instruction that is encoded with 16 bytes providing the
/// indices of the elements to return. The indices `i` in range [0, 15] select
/// the `i`-th element of `a`. The indices in range [16, 31] select the `i -
/// 16`-th element of `b`.
///
/// Note that this is a macro due to the codegen requirements of all of the
/// index expressions `$i*` must be constant. A compiler error will be
/// generated if any of the expressions are not constant.
///
/// All indexes `$i*` must have the type `u32`.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
    const I8: usize,
    const I9: usize,
    const I10: usize,
    const I11: usize,
    const I12: usize,
    const I13: usize,
    const I14: usize,
    const I15: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    let shuf = simd_shuffle16::<u8x16, u8x16>(
        a.as_u8x16(),
        b.as_u8x16(),
        [
            I0 as u32, I1 as u32, I2 as u32, I3 as u32, I4 as u32, I5 as u32, I6 as u32, I7 as u32,
            I8 as u32, I9 as u32, I10 as u32, I11 as u32, I12 as u32, I13 as u32, I14 as u32,
            I15 as u32,
        ],
    );
    transmute(shuf)
}

#[cfg(test)]
#[assert_instr(i8x16.shuffle)]
#[target_feature(enable = "simd128")]
unsafe fn i8x16_shuffle_test(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a, b)
}

/// Same as [`i8x16_shuffle`], except operates as if the inputs were eight
/// 16-bit integers, only taking 8 indices to shuffle.
///
/// Indices in the range [0, 7] select from `a` while [8, 15] select from `b`.
/// Note that this will generate the `i8x16.shuffle` instruction, since there
/// is no native `i16x8.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    let shuf = simd_shuffle8::<u16x8, u16x8>(
        a.as_u16x8(),
        b.as_u16x8(),
        [
            I0 as u32, I1 as u32, I2 as u32, I3 as u32, I4 as u32, I5 as u32, I6 as u32, I7 as u32,
        ],
    );
    transmute(shuf)
}

#[cfg(test)]
#[assert_instr(i8x16.shuffle)]
#[target_feature(enable = "simd128")]
unsafe fn i16x8_shuffle_test(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b)
}

/// Same as [`i8x16_shuffle`], except operates as if the inputs were four
/// 32-bit integers, only taking 4 indices to shuffle.
///
/// Indices in the range [0, 3] select from `a` while [4, 7] select from `b`.
/// Note that this will generate the `i8x16.shuffle` instruction, since there
/// is no native `i32x4.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: v128,
    b: v128,
) -> v128 {
    let shuf = simd_shuffle4::<u32x4, u32x4>(
        a.as_u32x4(),
        b.as_u32x4(),
        [I0 as u32, I1 as u32, I2 as u32, I3 as u32],
    );
    transmute(shuf)
}

#[cfg(test)]
#[assert_instr(i8x16.shuffle)]
#[target_feature(enable = "simd128")]
unsafe fn i32x4_shuffle_test(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 2, 4, 6>(a, b)
}

/// Same as [`i8x16_shuffle`], except operates as if the inputs were two
/// 64-bit integers, only taking 2 indices to shuffle.
///
/// Indices in the range [0, 1] select from `a` while [2, 3] select from `b`.
/// Note that this will generate the `v8x16.shuffle` instruction, since there
/// is no native `i64x2.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_shuffle<const I0: usize, const I1: usize>(a: v128, b: v128) -> v128 {
    let shuf = simd_shuffle2::<u64x2, u64x2>(a.as_u64x2(), b.as_u64x2(), [I0 as u32, I1 as u32]);
    transmute(shuf)
}

#[cfg(test)]
#[assert_instr(i8x16.shuffle)]
#[target_feature(enable = "simd128")]
unsafe fn i64x2_shuffle_test(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<0, 2>(a, b)
}

/// Extracts a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_extract_lane<const N: usize>(a: v128) -> i8 {
    simd_extract(a.as_i8x16(), N as u32)
}

#[cfg(test)]
#[assert_instr(i8x16.extract_lane_s)]
#[target_feature(enable = "simd128")]
unsafe fn i8x16_extract_lane_s(a: v128) -> i32 {
    i8x16_extract_lane::<0>(a) as i32
}

#[cfg(test)]
#[assert_instr(i8x16.extract_lane_u)]
#[target_feature(enable = "simd128")]
unsafe fn i8x16_extract_lane_u(a: v128) -> u32 {
    i8x16_extract_lane::<0>(a) as u8 as u32
}

/// Replaces a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_replace_lane<const N: usize>(a: v128, val: i8) -> v128 {
    transmute(simd_insert(a.as_i8x16(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(i8x16.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i8x16_replace_lane_test(a: v128, val: i8) -> v128 {
    i8x16_replace_lane::<0>(a, val)
}

/// Extracts a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Extracts a the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extract_lane<const N: usize>(a: v128) -> i16 {
    simd_extract(a.as_i16x8(), N as u32)
}

#[cfg(test)]
#[assert_instr(i16x8.extract_lane_s)]
#[target_feature(enable = "simd128")]
unsafe fn i16x8_extract_lane_s(a: v128) -> i32 {
    i16x8_extract_lane::<0>(a) as i32
}

#[cfg(test)]
#[assert_instr(i16x8.extract_lane_u)]
#[target_feature(enable = "simd128")]
unsafe fn i16x8_extract_lane_u(a: v128) -> u32 {
    i16x8_extract_lane::<0>(a) as u16 as u32
}

/// Replaces a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_replace_lane<const N: usize>(a: v128, val: i16) -> v128 {
    transmute(simd_insert(a.as_i16x8(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(i16x8.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i16x8_replace_lane_test(a: v128, val: i16) -> v128 {
    i16x8_replace_lane::<0>(a, val)
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extract_lane<const N: usize>(a: v128) -> i32 {
    simd_extract(a.as_i32x4(), N as u32)
}

#[cfg(test)]
#[assert_instr(i32x4.extract_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i32x4_extract_lane_test(a: v128) -> i32 {
    i32x4_extract_lane::<0>(a)
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_replace_lane<const N: usize>(a: v128, val: i32) -> v128 {
    transmute(simd_insert(a.as_i32x4(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(i32x4.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i32x4_replace_lane_test(a: v128, val: i32) -> v128 {
    i32x4_replace_lane::<0>(a, val)
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extract_lane<const N: usize>(a: v128) -> i64 {
    simd_extract(a.as_i64x2(), N as u32)
}

#[cfg(test)]
#[assert_instr(i64x2.extract_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i64x2_extract_lane_test(a: v128) -> i64 {
    i64x2_extract_lane::<0>(a)
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_replace_lane<const N: usize>(a: v128, val: i64) -> v128 {
    transmute(simd_insert(a.as_i64x2(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(i64x2.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn i64x2_replace_lane_test(a: v128, val: i64) -> v128 {
    i64x2_replace_lane::<0>(a, val)
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Extracts the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_extract_lane<const N: usize>(a: v128) -> f32 {
    simd_extract(a.as_f32x4(), N as u32)
}

#[cfg(test)]
#[assert_instr(f32x4.extract_lane)]
#[target_feature(enable = "simd128")]
unsafe fn f32x4_extract_lane_test(a: v128) -> f32 {
    f32x4_extract_lane::<0>(a)
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Replaces the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_replace_lane<const N: usize>(a: v128, val: f32) -> v128 {
    transmute(simd_insert(a.as_f32x4(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(f32x4.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn f32x4_replace_lane_test(a: v128, val: f32) -> v128 {
    f32x4_replace_lane::<0>(a, val)
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Extracts the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` fs out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_extract_lane<const N: usize>(a: v128) -> f64 {
    simd_extract(a.as_f64x2(), N as u32)
}

#[cfg(test)]
#[assert_instr(f64x2.extract_lane)]
#[target_feature(enable = "simd128")]
unsafe fn f64x2_extract_lane_test(a: v128) -> f64 {
    f64x2_extract_lane::<0>(a)
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_replace_lane<const N: usize>(a: v128, val: f64) -> v128 {
    transmute(simd_insert(a.as_f64x2(), N as u32, val))
}

#[cfg(test)]
#[assert_instr(f64x2.replace_lane)]
#[target_feature(enable = "simd128")]
unsafe fn f64x2_replace_lane_test(a: v128, val: f64) -> v128 {
    f64x2_replace_lane::<0>(a, val)
}

/// Returns a new vector with lanes selected from the lanes of the first input
/// vector `a` specified in the second input vector `s`.
///
/// The indices `i` in range [0, 15] select the `i`-th element of `a`. For
/// indices outside of the range the resulting lane is 0.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.swizzle))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_swizzle(a: v128, s: v128) -> v128 {
    transmute(llvm_swizzle(transmute(a), transmute(s)))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 16 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_splat(a: i8) -> v128 {
    transmute(i8x16::splat(a))
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 8 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_splat(a: i16) -> v128 {
    transmute(i16x8::splat(a))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_splat(a: i32) -> v128 {
    transmute(i32x4::splat(a))
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_splat(a: i64) -> v128 {
    transmute(i64x2::splat(a))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_splat(a: f32) -> v128 {
    transmute(f32x4::splat(a))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.splat))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_splat(a: f64) -> v128 {
    transmute(f64x2::splat(a))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.eq))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_eq(a: v128, b: v128) -> v128 {
    transmute(simd_eq::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ne))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_lt_s(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_lt_u(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i8x16>(a.as_u8x16(), b.as_u8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_gt_s(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_gt_u(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i8x16>(a.as_u8x16(), b.as_u8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_le_s(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_le_u(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i8x16>(a.as_u8x16(), b.as_u8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_ge_s(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i8x16>(a.as_i8x16(), b.as_i8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_ge_u(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i8x16>(a.as_u8x16(), b.as_u8x16()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.eq))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_eq(a: v128, b: v128) -> v128 {
    transmute(simd_eq::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ne))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_lt_s(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_lt_u(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i16x8>(a.as_u16x8(), b.as_u16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_gt_s(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_gt_u(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i16x8>(a.as_u16x8(), b.as_u16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_le_s(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_le_u(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i16x8>(a.as_u16x8(), b.as_u16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_ge_s(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i16x8>(a.as_i16x8(), b.as_i16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_ge_u(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i16x8>(a.as_u16x8(), b.as_u16x8()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.eq))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_eq(a: v128, b: v128) -> v128 {
    transmute(simd_eq::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ne))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_lt_s(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_lt_u(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i32x4>(a.as_u32x4(), b.as_u32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_gt_s(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_gt_u(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i32x4>(a.as_u32x4(), b.as_u32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_le_s(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_le_u(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i32x4>(a.as_u32x4(), b.as_u32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_ge_s(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i32x4>(a.as_i32x4(), b.as_i32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_ge_u(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i32x4>(a.as_u32x4(), b.as_u32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.eq))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_eq(a: v128, b: v128) -> v128 {
    transmute(llvm_eq(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.ne))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.lt_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_lt_s(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.gt_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_gt_s(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.le_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_le_s(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.ge_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_ge_s(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.eq))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_eq(a: v128, b: v128) -> v128 {
    transmute(simd_eq::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ne))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.lt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_lt(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.gt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_gt(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.le))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_le(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ge))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_ge(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i32x4>(a.as_f32x4(), b.as_f32x4()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.eq))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_eq(a: v128, b: v128) -> v128 {
    transmute(simd_eq::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ne))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_ne(a: v128, b: v128) -> v128 {
    transmute(simd_ne::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.lt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_lt(a: v128, b: v128) -> v128 {
    transmute(simd_lt::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.gt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_gt(a: v128, b: v128) -> v128 {
    transmute(simd_gt::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.le))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_le(a: v128, b: v128) -> v128 {
    transmute(simd_le::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ge))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_ge(a: v128, b: v128) -> v128 {
    transmute(simd_ge::<_, i64x2>(a.as_f64x2(), b.as_f64x2()))
}

/// Flips each bit of the 128-bit input vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.not))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_not(a: v128) -> v128 {
    transmute(simd_xor(a.as_i64x2(), i64x2(!0, !0)))
}

/// Performs a bitwise and of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.and))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_and(a: v128, b: v128) -> v128 {
    transmute(simd_and(a.as_i64x2(), b.as_i64x2()))
}

/// Bitwise AND of bits of `a` and the logical inverse of bits of `b`.
///
/// This operation is equivalent to `v128.and(a, v128.not(b))`
#[inline]
#[cfg_attr(test, assert_instr(v128.andnot))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_andnot(a: v128, b: v128) -> v128 {
    transmute(simd_and(
        a.as_i64x2(),
        simd_xor(b.as_i64x2(), i64x2(-1, -1)),
    ))
}

/// Performs a bitwise or of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.or))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_or(a: v128, b: v128) -> v128 {
    transmute(simd_or(a.as_i64x2(), b.as_i64x2()))
}

/// Performs a bitwise xor of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.xor))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_xor(a: v128, b: v128) -> v128 {
    transmute(simd_xor(a.as_i64x2(), b.as_i64x2()))
}

/// Use the bitmask in `c` to select bits from `v1` when 1 and `v2` when 0.
#[inline]
#[cfg_attr(test, assert_instr(v128.bitselect))]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    transmute(llvm_bitselect(v1.as_i8x16(), v2.as_i8x16(), c.as_i8x16()))
}

/// Returns true if any lane is nonzero or false if all lanes are zero.
#[inline]
// #[cfg_attr(test, assert_instr(v128.any_true))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn v128_any_true(a: v128) -> bool {
    llvm_any_true_i8x16(a.as_i8x16()) != 0
}

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.abs))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_abs(a: v128) -> v128 {
    let a = transmute::<_, i8x16>(a);
    let zero = i8x16::splat(0);
    transmute(simd_select::<m8x16, i8x16>(
        simd_lt(a, zero),
        simd_sub(zero, a),
        a,
    ))
}

/// Negates a 128-bit vectors intepreted as sixteen 8-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i8x16.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_neg(a: v128) -> v128 {
    transmute(simd_mul(a.as_i8x16(), i8x16::splat(-1)))
}

/// Count the number of bits set to one within each lane.
#[inline]
// #[cfg_attr(test, assert_instr(i8x16.popcnt))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_popcnt(v: v128) -> v128 {
    transmute(llvm_popcnt(v.as_i8x16()))
}

/// Returns true if all lanes are nonzero or false if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.all_true))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_all_true(a: v128) -> bool {
    llvm_i8x16_all_true(a.as_i8x16()) != 0
}

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
// #[cfg_attr(test, assert_instr(i8x16.bitmask))] // FIXME
pub unsafe fn i8x16_bitmask(a: v128) -> i32 {
    llvm_bitmask_i8x16(transmute(a))
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x7f or 0x80 is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.narrow_i16x8_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_narrow_i16x8_s(a: v128, b: v128) -> v128 {
    transmute(llvm_narrow_i8x16_s(transmute(a), transmute(b)))
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x00 or 0xff is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.narrow_i16x8_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_narrow_i16x8_u(a: v128, b: v128) -> v128 {
    transmute(llvm_narrow_i8x16_u(transmute(a), transmute(b)))
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shl))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_shl(a: v128, amt: u32) -> v128 {
    transmute(simd_shl(a.as_i8x16(), i8x16::splat(amt as i8)))
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shr_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_shr_s(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_i8x16(), i8x16::splat(amt as i8)))
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_shr_u(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_u8x16(), u8x16::splat(amt as u8)))
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_i8x16(), b.as_i8x16()))
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit signed
/// integers, saturating on overflow to `i8::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_sat_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_add_sat_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i8x16_add_sat_s(a.as_i8x16(), b.as_i8x16()))
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit unsigned
/// integers, saturating on overflow to `u8::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_sat_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_add_sat_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i8x16_add_sat_u(a.as_i8x16(), b.as_i8x16()))
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_i8x16(), b.as_i8x16()))
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// signed integers, saturating on overflow to `i8::MIN`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_sat_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_sub_sat_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i8x16_sub_sat_s(a.as_i8x16(), b.as_i8x16()))
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_sat_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_sub_sat_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i8x16_sub_sat_u(a.as_i8x16(), b.as_i8x16()))
}

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.min_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_min_s(a: v128, b: v128) -> v128 {
    let a = a.as_i8x16();
    let b = b.as_i8x16();
    transmute(simd_select::<i8x16, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.min_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_min_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u8x16>(a);
    let b = transmute::<_, u8x16>(b);
    transmute(simd_select::<i8x16, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.max_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_max_s(a: v128, b: v128) -> v128 {
    let a = transmute::<_, i8x16>(a);
    let b = transmute::<_, i8x16>(b);
    transmute(simd_select::<i8x16, _>(simd_gt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.max_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_max_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u8x16>(a);
    let b = transmute::<_, u8x16>(b);
    transmute(simd_select::<i8x16, _>(simd_gt(a, b), a, b))
}

/// Lane-wise rounding average.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.avgr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i8x16_avgr_u(a: v128, b: v128) -> v128 {
    transmute(llvm_avgr_u_i8x16(transmute(a), transmute(b)))
}

/// Lane-wise integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.extadd_pairwise_i8x16_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extadd_pairwise_i8x16_s(a: v128) -> v128 {
    transmute(llvm_i16x8_extadd_pairwise_i8x16_s(a.as_i8x16()))
}

/// Lane-wise integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.extadd_pairwise_i8x16_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extadd_pairwise_i8x16_u(a: v128) -> v128 {
    transmute(llvm_i16x8_extadd_pairwise_i8x16_u(a.as_i8x16()))
}

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.abs))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_abs(a: v128) -> v128 {
    let a = transmute::<_, i16x8>(a);
    let zero = i16x8::splat(0);
    transmute(simd_select::<m16x8, i16x8>(
        simd_lt(a, zero),
        simd_sub(zero, a),
        a,
    ))
}

/// Negates a 128-bit vectors intepreted as eight 16-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i16x8.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_neg(a: v128) -> v128 {
    transmute(simd_mul(a.as_i16x8(), i16x8::splat(-1)))
}

/// Lane-wise saturating rounding multiplication in Q15 format.
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.qmulr_sat_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_q15mulr_sat_s(a: v128, b: v128) -> v128 {
    transmute(llvm_q15mulr(a.as_i16x8(), b.as_i16x8()))
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.all_true))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_all_true(a: v128) -> bool {
    llvm_i16x8_all_true(a.as_i16x8()) != 0
}

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.bitmask))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_bitmask(a: v128) -> i32 {
    llvm_bitmask_i16x8(transmute(a))
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x7fff or 0x8000 is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.narrow_i32x4_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_narrow_i32x4_s(a: v128, b: v128) -> v128 {
    transmute(llvm_narrow_i16x8_s(transmute(a), transmute(b)))
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x0000 or 0xffff is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.narrow_i32x4_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_narrow_i32x4_u(a: v128, b: v128) -> v128 {
    transmute(llvm_narrow_i16x8_u(transmute(a), transmute(b)))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_low_i8x16_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extend_low_i8x16_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i16x8>(simd_shuffle8::<_, i8x8>(
        a.as_i8x16(),
        a.as_i8x16(),
        [0, 1, 2, 3, 4, 5, 6, 7],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_high_i8x16_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extend_high_i8x16_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i16x8>(simd_shuffle8::<_, i8x8>(
        a.as_i8x16(),
        a.as_i8x16(),
        [8, 9, 10, 11, 12, 13, 14, 15],
    )))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_low_i8x16_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extend_low_i8x16_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u16x8>(simd_shuffle8::<_, u8x8>(
        a.as_u8x16(),
        a.as_u8x16(),
        [0, 1, 2, 3, 4, 5, 6, 7],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_high_i8x16_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extend_high_i8x16_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u16x8>(simd_shuffle8::<_, u8x8>(
        a.as_u8x16(),
        a.as_u8x16(),
        [8, 9, 10, 11, 12, 13, 14, 15],
    )))
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shl))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_shl(a: v128, amt: u32) -> v128 {
    transmute(simd_shl(a.as_i16x8(), i16x8::splat(amt as i16)))
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shr_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_shr_s(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_i16x8(), i16x8::splat(amt as i16)))
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_shr_u(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_u16x8(), u16x8::splat(amt as u16)))
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_i16x8(), b.as_i16x8()))
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit signed
/// integers, saturating on overflow to `i16::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_sat_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_add_sat_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_add_sat_s(a.as_i16x8(), b.as_i16x8()))
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit unsigned
/// integers, saturating on overflow to `u16::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_sat_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_add_sat_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_add_sat_u(a.as_i16x8(), b.as_i16x8()))
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_i16x8(), b.as_i16x8()))
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers, saturating on overflow to `i16::MIN`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_sat_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_sub_sat_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_sub_sat_s(a.as_i16x8(), b.as_i16x8()))
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_sat_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_sub_sat_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_sub_sat_u(a.as_i16x8(), b.as_i16x8()))
}

/// Multiplies two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.mul))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_mul(a: v128, b: v128) -> v128 {
    transmute(simd_mul(a.as_i16x8(), b.as_i16x8()))
}

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.min_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_min_s(a: v128, b: v128) -> v128 {
    let a = transmute::<_, i16x8>(a);
    let b = transmute::<_, i16x8>(b);
    transmute(simd_select::<i16x8, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.min_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_min_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u16x8>(a);
    let b = transmute::<_, u16x8>(b);
    transmute(simd_select::<i16x8, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.max_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_max_s(a: v128, b: v128) -> v128 {
    let a = transmute::<_, i16x8>(a);
    let b = transmute::<_, i16x8>(b);
    transmute(simd_select::<i16x8, _>(simd_gt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.max_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_max_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u16x8>(a);
    let b = transmute::<_, u16x8>(b);
    transmute(simd_select::<i16x8, _>(simd_gt(a, b), a, b))
}

/// Lane-wise rounding average.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.avgr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_avgr_u(a: v128, b: v128) -> v128 {
    transmute(llvm_avgr_u_i16x8(transmute(a), transmute(b)))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_low_i8x16_s(a), i16x8_extend_low_i8x16_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.avgr_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extmul_low_i8x16_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_extmul_low_i8x16_s(a.as_i8x16(), b.as_i8x16()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_high_i8x16_s(a), i16x8_extend_high_i8x16_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.avgr_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extmul_high_i8x16_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_extmul_high_i8x16_s(a.as_i8x16(), b.as_i8x16()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_low_i8x16_u(a), i16x8_extend_low_i8x16_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.avgr_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extmul_low_i8x16_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_extmul_low_i8x16_u(a.as_i8x16(), b.as_i8x16()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_high_i8x16_u(a), i16x8_extend_high_i8x16_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i16x8.avgr_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i16x8_extmul_high_i8x16_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i16x8_extmul_high_i8x16_u(a.as_i8x16(), b.as_i8x16()))
}

/// Lane-wise integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extadd_pairwise_i16x8_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extadd_pairwise_i16x8_s(a: v128) -> v128 {
    transmute(llvm_i32x4_extadd_pairwise_i16x8_s(a.as_i16x8()))
}

/// Lane-wise integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extadd_pairwise_i16x8_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extadd_pairwise_i16x8_u(a: v128) -> v128 {
    transmute(llvm_i32x4_extadd_pairwise_i16x8_u(a.as_i16x8()))
}

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.abs))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_abs(a: v128) -> v128 {
    let a = transmute::<_, i32x4>(a);
    let zero = i32x4::splat(0);
    transmute(simd_select::<m32x4, i32x4>(
        simd_lt(a, zero),
        simd_sub(zero, a),
        a,
    ))
}

/// Negates a 128-bit vectors intepreted as four 32-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i32x4.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_neg(a: v128) -> v128 {
    transmute(simd_mul(a.as_i32x4(), i32x4::splat(-1)))
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.all_true))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_all_true(a: v128) -> bool {
    llvm_i32x4_all_true(a.as_i32x4()) != 0
}

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.bitmask))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_bitmask(a: v128) -> i32 {
    llvm_bitmask_i32x4(transmute(a))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_low_i16x8_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extend_low_i16x8_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i32x4>(simd_shuffle4::<_, i16x4>(
        a.as_i16x8(),
        a.as_i16x8(),
        [0, 1, 2, 3],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_high_i16x8_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extend_high_i16x8_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i32x4>(simd_shuffle4::<_, i16x4>(
        a.as_i16x8(),
        a.as_i16x8(),
        [4, 5, 6, 7],
    )))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_low_i16x8_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extend_low_i16x8_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u32x4>(simd_shuffle4::<_, u16x4>(
        a.as_u16x8(),
        a.as_u16x8(),
        [0, 1, 2, 3],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_high_i16x8_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extend_high_i16x8_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u32x4>(simd_shuffle4::<_, u16x4>(
        a.as_u16x8(),
        a.as_u16x8(),
        [4, 5, 6, 7],
    )))
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shl))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_shl(a: v128, amt: u32) -> v128 {
    transmute(simd_shl(a.as_i32x4(), i32x4::splat(amt as i32)))
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shr_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_shr_s(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_i32x4(), i32x4::splat(amt as i32)))
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_shr_u(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_u32x4(), u32x4::splat(amt as u32)))
}

/// Adds two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_i32x4(), b.as_i32x4()))
}

/// Subtracts two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_i32x4(), b.as_i32x4()))
}

/// Multiplies two 128-bit vectors as if they were two packed four 32-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.mul))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_mul(a: v128, b: v128) -> v128 {
    transmute(simd_mul(a.as_i32x4(), b.as_i32x4()))
}

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.min_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_min_s(a: v128, b: v128) -> v128 {
    let a = transmute::<_, i32x4>(a);
    let b = transmute::<_, i32x4>(b);
    transmute(simd_select::<i32x4, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.min_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_min_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u32x4>(a);
    let b = transmute::<_, u32x4>(b);
    transmute(simd_select::<i32x4, _>(simd_lt(a, b), a, b))
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.max_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_max_s(a: v128, b: v128) -> v128 {
    let a = transmute::<_, i32x4>(a);
    let b = transmute::<_, i32x4>(b);
    transmute(simd_select::<i32x4, _>(simd_gt(a, b), a, b))
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.max_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_max_u(a: v128, b: v128) -> v128 {
    let a = transmute::<_, u32x4>(a);
    let b = transmute::<_, u32x4>(b);
    transmute(simd_select::<i32x4, _>(simd_gt(a, b), a, b))
}

/// Lane-wise multiply signed 16-bit integers in the two input vectors and add
/// adjacent pairs of the full 32-bit results.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.dot_i16x8_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_dot_i16x8_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i32x4_dot_i16x8_s(a.as_i16x8(), b.as_i16x8()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_low_i16x8_s(a), i32x4_extend_low_i16x8_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extmul_low_i16x8_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extmul_low_i16x8_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i32x4_extmul_low_i16x8_s(a.as_i16x8(), b.as_i16x8()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_high_i16x8_s(a), i32x4_extend_high_i16x8_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extmul_high_i16x8_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extmul_high_i16x8_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i32x4_extmul_high_i16x8_s(a.as_i16x8(), b.as_i16x8()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_low_i16x8_u(a), i32x4_extend_low_i16x8_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extmul_low_i16x8_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extmul_low_i16x8_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i32x4_extmul_low_i16x8_u(a.as_i16x8(), b.as_i16x8()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_high_i16x8_u(a), i32x4_extend_high_i16x8_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.extmul_high_i16x8_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_extmul_high_i16x8_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i32x4_extmul_high_i16x8_u(a.as_i16x8(), b.as_i16x8()))
}

/// Lane-wise wrapping absolute value.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.abs))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_abs(a: v128) -> v128 {
    let a = transmute::<_, i64x2>(a);
    let zero = i64x2::splat(0);
    transmute(simd_select::<m64x2, i64x2>(
        simd_lt(a, zero),
        simd_sub(zero, a),
        a,
    ))
}

/// Negates a 128-bit vectors intepreted as two 64-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i64x2.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_neg(a: v128) -> v128 {
    transmute(simd_mul(a.as_i64x2(), i64x2::splat(-1)))
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.all_true))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_all_true(a: v128) -> bool {
    llvm_i64x2_all_true(a.as_i64x2()) != 0
}

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.bitmask))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_bitmask(a: v128) -> i32 {
    llvm_bitmask_i64x2(transmute(a))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extend_low_i32x4_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extend_low_i32x4_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i64x2>(simd_shuffle2::<_, i32x2>(
        a.as_i32x4(),
        a.as_i32x4(),
        [0, 1],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extend_high_i32x4_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extend_high_i32x4_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i64x2>(simd_shuffle2::<_, i32x2>(
        a.as_i32x4(),
        a.as_i32x4(),
        [2, 3],
    )))
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extend_low_i32x4_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extend_low_i32x4_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u64x2>(simd_shuffle2::<_, u32x2>(
        a.as_u32x4(),
        a.as_u32x4(),
        [0, 1],
    )))
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extend_high_i32x4_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extend_high_i32x4_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u64x2>(simd_shuffle2::<_, u32x2>(
        a.as_u32x4(),
        a.as_u32x4(),
        [2, 3],
    )))
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shl))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_shl(a: v128, amt: u32) -> v128 {
    transmute(simd_shl(a.as_i64x2(), i64x2::splat(amt as i64)))
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shr_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_shr_s(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_i64x2(), i64x2::splat(amt as i64)))
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shr_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_shr_u(a: v128, amt: u32) -> v128 {
    transmute(simd_shr(a.as_u64x2(), u64x2::splat(amt as u64)))
}

/// Adds two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_i64x2(), b.as_i64x2()))
}

/// Subtracts two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_i64x2(), b.as_i64x2()))
}

/// Multiplies two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.mul))]
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_mul(a: v128, b: v128) -> v128 {
    transmute(simd_mul(a.as_i64x2(), b.as_i64x2()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_low_i32x4_s(a), i64x2_extend_low_i32x4_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extmul_low_i32x4_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extmul_low_i32x4_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i64x2_extmul_low_i32x4_s(a.as_i32x4(), b.as_i32x4()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_high_i32x4_s(a), i64x2_extend_high_i32x4_s(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extmul_high_i32x4_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extmul_high_i32x4_s(a: v128, b: v128) -> v128 {
    transmute(llvm_i64x2_extmul_high_i32x4_s(a.as_i32x4(), b.as_i32x4()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_low_i32x4_u(a), i64x2_extend_low_i32x4_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extmul_low_i32x4_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extmul_low_i32x4_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i64x2_extmul_low_i32x4_u(a.as_i32x4(), b.as_i32x4()))
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_high_i32x4_u(a), i64x2_extend_high_i32x4_u(b))`
#[inline]
// #[cfg_attr(test, assert_instr(i64x2.extmul_high_i32x4_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i64x2_extmul_high_i32x4_u(a: v128, b: v128) -> v128 {
    transmute(llvm_i64x2_extmul_high_i32x4_u(a.as_i32x4(), b.as_i32x4()))
}

/// Lane-wise rounding to the nearest integral value not smaller than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.ceil))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_ceil(a: v128) -> v128 {
    transmute(llvm_f32x4_ceil(a.as_f32x4()))
}

/// Lane-wise rounding to the nearest integral value not greater than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.floor))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_floor(a: v128) -> v128 {
    transmute(llvm_f32x4_floor(a.as_f32x4()))
}

/// Lane-wise rounding to the nearest integral value with the magnitude not
/// larger than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.trunc))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_trunc(a: v128) -> v128 {
    transmute(llvm_f32x4_trunc(a.as_f32x4()))
}

/// Lane-wise rounding to the nearest integral value; if two values are equally
/// near, rounds to the even one.
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.nearest))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_nearest(a: v128) -> v128 {
    transmute(llvm_f32x4_nearest(a.as_f32x4()))
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.abs))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_abs(a: v128) -> v128 {
    transmute(llvm_f32x4_abs(a.as_f32x4()))
}

/// Negates each lane of a 128-bit vector interpreted as four 32-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_neg(a: v128) -> v128 {
    f32x4_mul(a, f32x4_splat(-1.))
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sqrt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_sqrt(a: v128) -> v128 {
    transmute(llvm_f32x4_sqrt(a.as_f32x4()))
}

/// Adds pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_f32x4(), b.as_f32x4()))
}

/// Subtracts pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_f32x4(), b.as_f32x4()))
}

/// Multiplies pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.mul))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_mul(a: v128, b: v128) -> v128 {
    transmute(simd_mul(a.as_f32x4(), b.as_f32x4()))
}

/// Divides pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.div))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_div(a: v128, b: v128) -> v128 {
    transmute(simd_div(a.as_f32x4(), b.as_f32x4()))
}

/// Calculates the minimum of pairwise lanes of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.min))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_min(a: v128, b: v128) -> v128 {
    transmute(llvm_f32x4_min(a.as_f32x4(), b.as_f32x4()))
}

/// Calculates the maximum of pairwise lanes of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.max))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_max(a: v128, b: v128) -> v128 {
    transmute(llvm_f32x4_max(a.as_f32x4(), b.as_f32x4()))
}

/// Lane-wise minimum value, defined as `b < a ? b : a`
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.pmin))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_pmin(a: v128, b: v128) -> v128 {
    transmute(llvm_f32x4_pmin(a.as_f32x4(), b.as_f32x4()))
}

/// Lane-wise maximum value, defined as `a < b ? b : a`
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.pmax))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_pmax(a: v128, b: v128) -> v128 {
    transmute(llvm_f32x4_pmax(a.as_f32x4(), b.as_f32x4()))
}

/// Lane-wise rounding to the nearest integral value not smaller than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.ceil))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_ceil(a: v128) -> v128 {
    transmute(llvm_f64x2_ceil(a.as_f64x2()))
}

/// Lane-wise rounding to the nearest integral value not greater than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.floor))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_floor(a: v128) -> v128 {
    transmute(llvm_f64x2_floor(a.as_f64x2()))
}

/// Lane-wise rounding to the nearest integral value with the magnitude not
/// larger than the input.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.trunc))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_trunc(a: v128) -> v128 {
    transmute(llvm_f64x2_trunc(a.as_f64x2()))
}

/// Lane-wise rounding to the nearest integral value; if two values are equally
/// near, rounds to the even one.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.nearest))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_nearest(a: v128) -> v128 {
    transmute(llvm_f64x2_nearest(a.as_f64x2()))
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.abs))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_abs(a: v128) -> v128 {
    transmute(llvm_f64x2_abs(a.as_f64x2()))
}

/// Negates each lane of a 128-bit vector interpreted as two 64-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.neg))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_neg(a: v128) -> v128 {
    f64x2_mul(a, f64x2_splat(-1.0))
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sqrt))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_sqrt(a: v128) -> v128 {
    transmute(llvm_f64x2_sqrt(a.as_f64x2()))
}

/// Adds pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.add))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_add(a: v128, b: v128) -> v128 {
    transmute(simd_add(a.as_f64x2(), b.as_f64x2()))
}

/// Subtracts pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sub))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_sub(a: v128, b: v128) -> v128 {
    transmute(simd_sub(a.as_f64x2(), b.as_f64x2()))
}

/// Multiplies pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.mul))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_mul(a: v128, b: v128) -> v128 {
    transmute(simd_mul(a.as_f64x2(), b.as_f64x2()))
}

/// Divides pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.div))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_div(a: v128, b: v128) -> v128 {
    transmute(simd_div(a.as_f64x2(), b.as_f64x2()))
}

/// Calculates the minimum of pairwise lanes of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.min))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_min(a: v128, b: v128) -> v128 {
    transmute(llvm_f64x2_min(a.as_f64x2(), b.as_f64x2()))
}

/// Calculates the maximum of pairwise lanes of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.max))]
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_max(a: v128, b: v128) -> v128 {
    transmute(llvm_f64x2_max(a.as_f64x2(), b.as_f64x2()))
}

/// Lane-wise minimum value, defined as `b < a ? b : a`
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.pmin))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_pmin(a: v128, b: v128) -> v128 {
    transmute(llvm_f64x2_pmin(a.as_f64x2(), b.as_f64x2()))
}

/// Lane-wise maximum value, defined as `a < b ? b : a`
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.pmax))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_pmax(a: v128, b: v128) -> v128 {
    transmute(llvm_f64x2_pmax(a.as_f64x2(), b.as_f64x2()))
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit signed integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f32x4_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_trunc_sat_f32x4_s(a: v128) -> v128 {
    transmute(simd_cast::<_, i32x4>(a.as_f32x4()))
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit unsigned integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f32x4_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_trunc_sat_f32x4_u(a: v128) -> v128 {
    transmute(simd_cast::<_, u32x4>(a.as_f32x4()))
}

/// Converts a 128-bit vector interpreted as four 32-bit signed integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.convert_i32x4_s))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_convert_i32x4_s(a: v128) -> v128 {
    transmute(simd_cast::<_, f32x4>(a.as_i32x4()))
}

/// Converts a 128-bit vector interpreted as four 32-bit unsigned integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.convert_i32x4_u))]
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_convert_i32x4_u(a: v128) -> v128 {
    transmute(simd_cast::<_, f32x4>(a.as_u32x4()))
}

/// Saturating conversion of the two double-precision floating point lanes to
/// two lower integer lanes using the IEEE `convertToIntegerTowardZero`
/// function.
///
/// The two higher lanes of the result are initialized to zero. If any input
/// lane is a NaN, the resulting lane is 0. If the rounded integer value of a
/// lane is outside the range of the destination type, the result is saturated
/// to the nearest representable integer value.
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.trunc_sat_f64x2_s_zero))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_trunc_sat_f64x2_s_zero(a: v128) -> v128 {
    transmute(llvm_i32x4_trunc_sat_f64x2_s_zero(a.as_f64x2()))
}

/// Saturating conversion of the two double-precision floating point lanes to
/// two lower integer lanes using the IEEE `convertToIntegerTowardZero`
/// function.
///
/// The two higher lanes of the result are initialized to zero. If any input
/// lane is a NaN, the resulting lane is 0. If the rounded integer value of a
/// lane is outside the range of the destination type, the result is saturated
/// to the nearest representable integer value.
#[inline]
// #[cfg_attr(test, assert_instr(i32x4.trunc_sat_f64x2_u_zero))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn i32x4_trunc_sat_f64x2_u_zero(a: v128) -> v128 {
    transmute(llvm_i32x4_trunc_sat_f64x2_u_zero(a.as_f64x2()))
}

/// Lane-wise conversion from integer to floating point.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.convert_low_i32x4_s))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_convert_low_i32x4_s(a: v128) -> v128 {
    transmute(llvm_f64x2_convert_low_i32x4_s(a.as_i32x4()))
}

/// Lane-wise conversion from integer to floating point.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.convert_low_i32x4_u))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_convert_low_i32x4_u(a: v128) -> v128 {
    transmute(llvm_f64x2_convert_low_i32x4_u(a.as_i32x4()))
}

/// Conversion of the two double-precision floating point lanes to two lower
/// single-precision lanes of the result. The two higher lanes of the result are
/// initialized to zero. If the conversion result is not representable as a
/// single-precision floating point number, it is rounded to the nearest-even
/// representable number.
#[inline]
// #[cfg_attr(test, assert_instr(f32x4.demote_f64x2_zero))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f32x4_demote_f64x2_zero(a: v128) -> v128 {
    transmute(llvm_f32x4_demote_f64x2_zero(a.as_f64x2()))
}

/// Conversion of the two lower single-precision floating point lanes to the two
/// double-precision lanes of the result.
#[inline]
// #[cfg_attr(test, assert_instr(f64x2.promote_low_f32x4))] // FIXME
#[target_feature(enable = "simd128")]
pub unsafe fn f64x2_promote_low_f32x4(a: v128) -> v128 {
    transmute(llvm_f64x2_promote_low_f32x4(a.as_f32x4()))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std;
    use std::mem;
    use std::num::Wrapping;
    use std::prelude::v1::*;

    fn compare_bytes(a: v128, b: v128) {
        let a: [u8; 16] = unsafe { transmute(a) };
        let b: [u8; 16] = unsafe { transmute(b) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_load() {
        unsafe {
            let arr: [i32; 4] = [0, 1, 2, 3];
            let vec = v128_load(arr.as_ptr() as *const v128);
            compare_bytes(vec, i32x4_const(0, 1, 2, 3));
        }
    }

    #[test]
    fn test_load_extend() {
        unsafe {
            let arr: [i8; 8] = [-3, -2, -1, 0, 1, 2, 3, 4];
            let vec = v128_load8x8_s(arr.as_ptr());
            compare_bytes(vec, i16x8_const(-3, -2, -1, 0, 1, 2, 3, 4));
            let vec = v128_load8x8_u(arr.as_ptr() as *const u8);
            compare_bytes(vec, i16x8_const(253, 254, 255, 0, 1, 2, 3, 4));

            let arr: [i16; 4] = [-1, 0, 1, 2];
            let vec = v128_load16x4_s(arr.as_ptr());
            compare_bytes(vec, i32x4_const(-1, 0, 1, 2));
            let vec = v128_load16x4_u(arr.as_ptr() as *const u16);
            compare_bytes(vec, i32x4_const(65535, 0, 1, 2));

            let arr: [i32; 2] = [-1, 1];
            let vec = v128_load32x2_s(arr.as_ptr());
            compare_bytes(vec, i64x2_const(-1, 1));
            let vec = v128_load32x2_u(arr.as_ptr() as *const u32);
            compare_bytes(vec, i64x2_const(u32::max_value().into(), 1));
        }
    }

    #[test]
    fn test_load_splat() {
        unsafe {
            compare_bytes(v128_load8_splat(&8), i8x16_splat(8));
            compare_bytes(v128_load16_splat(&9), i16x8_splat(9));
            compare_bytes(v128_load32_splat(&10), i32x4_splat(10));
            compare_bytes(v128_load64_splat(&11), i64x2_splat(11));
        }
    }

    // TODO: v128_load{32,64}_zero

    #[test]
    fn test_store() {
        unsafe {
            let mut spot = i8x16_splat(0);
            v128_store(&mut spot, i8x16_splat(1));
            compare_bytes(spot, i8x16_splat(1));
        }
    }

    // TODO: v128_load*_lane
    // TODO: v128_store*_lane

    #[test]
    fn test_v128_const() {
        const A: v128 =
            unsafe { super::v128_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) };
        compare_bytes(A, A);
    }

    #[test]
    fn test_shuffle() {
        unsafe {
            let vec_a = v128_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let vec_b = v128_const(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );

            let vec_r = i8x16_shuffle::<0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30>(
                vec_a, vec_b,
            );
            let vec_e = v128_const(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
            compare_bytes(vec_r, vec_e);

            let vec_a = i16x8_const(0, 1, 2, 3, 4, 5, 6, 7);
            let vec_b = i16x8_const(8, 9, 10, 11, 12, 13, 14, 15);
            let vec_r = i16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(vec_a, vec_b);
            let vec_e = i16x8_const(0, 8, 2, 10, 4, 12, 6, 14);
            compare_bytes(vec_r, vec_e);

            let vec_a = i32x4_const(0, 1, 2, 3);
            let vec_b = i32x4_const(4, 5, 6, 7);
            let vec_r = i32x4_shuffle::<0, 4, 2, 6>(vec_a, vec_b);
            let vec_e = i32x4_const(0, 4, 2, 6);
            compare_bytes(vec_r, vec_e);

            let vec_a = i64x2_const(0, 1);
            let vec_b = i64x2_const(2, 3);
            let vec_r = i64x2_shuffle::<0, 2>(vec_a, vec_b);
            let vec_e = i64x2_const(0, 2);
            compare_bytes(vec_r, vec_e);
        }
    }

    // tests extract and replace lanes
    macro_rules! test_extract {
        (
            name: $test_id:ident,
            extract: $extract:ident,
            replace: $replace:ident,
            elem: $elem:ty,
            count: $count:expr,
            indices: [$($idx:expr),*],
        ) => {
            #[test]
            fn $test_id() {
                unsafe {
                    let arr: [$elem; $count] = [123 as $elem; $count];
                    let vec: v128 = transmute(arr);
                    $(
                        assert_eq!($extract::<$idx>(vec), 123 as $elem);
                    )*

                    // create a vector from array and check that the indices contain
                    // the same values as in the array:
                    let arr: [$elem; $count] = [$($idx as $elem),*];
                    let vec: v128 = transmute(arr);
                    $(
                        assert_eq!($extract::<$idx>(vec), $idx as $elem);

                        let tmp = $replace::<$idx>(vec, 124 as $elem);
                        assert_eq!($extract::<$idx>(tmp), 124 as $elem);
                    )*
                }
            }
        }
    }

    test_extract! {
        name: test_i8x16_extract_replace,
        extract: i8x16_extract_lane,
        replace: i8x16_replace_lane,
        elem: i8,
        count: 16,
        indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    }
    test_extract! {
        name: test_i16x8_extract_replace,
        extract: i16x8_extract_lane,
        replace: i16x8_replace_lane,
        elem: i16,
        count: 8,
        indices: [0, 1, 2, 3, 4, 5, 6, 7],
    }
    test_extract! {
        name: test_i32x4_extract_replace,
        extract: i32x4_extract_lane,
        replace: i32x4_replace_lane,
        elem: i32,
        count: 4,
        indices: [0, 1, 2, 3],
    }
    test_extract! {
        name: test_i64x2_extract_replace,
        extract: i64x2_extract_lane,
        replace: i64x2_replace_lane,
        elem: i64,
        count: 2,
        indices: [0, 1],
    }
    test_extract! {
        name: test_f32x4_extract_replace,
        extract: f32x4_extract_lane,
        replace: f32x4_replace_lane,
        elem: f32,
        count: 4,
        indices: [0, 1, 2, 3],
    }
    test_extract! {
        name: test_f64x2_extract_replace,
        extract: f64x2_extract_lane,
        replace: f64x2_replace_lane,
        elem: f64,
        count: 2,
        indices: [0, 1],
    }

    #[test]
    #[rustfmt::skip]
    fn test_swizzle() {
        unsafe {
            compare_bytes(
                i8x16_swizzle(
                    i32x4_const(1, 2, 3, 4),
                    v128_const(
                        32, 31, 30, 29,
                        0, 1, 2, 3,
                        12, 13, 14, 15,
                        0, 4, 8, 12),
                ),
                i32x4_const(0, 1, 4, 0x04030201),
            );
        }
    }

    macro_rules! test_splat {
        ($test_id:ident: $val:expr => $($vals:expr),*) => {
            #[test]
            fn $test_id() {
                unsafe {
                    let a = super::$test_id($val);
                    let b: v128 = transmute([$($vals as u8),*]);
                    compare_bytes(a, b);
                }
            }
        }
    }

    mod splats {
        use super::*;
        test_splat!(i8x16_splat: 42 => 42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42);
        test_splat!(i16x8_splat: 42 => 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0);
        test_splat!(i32x4_splat: 42 => 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0);
        test_splat!(i64x2_splat: 42 => 42, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0);
        test_splat!(f32x4_splat: 42. => 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66);
        test_splat!(f64x2_splat: 42. => 0, 0, 0, 0, 0, 0, 69, 64, 0, 0, 0, 0, 0, 0, 69, 64);
    }

    macro_rules! test_binop {
        (
            $($name:ident => {
                $([$($vec1:tt)*] ($op:tt | $f:ident) [$($vec2:tt)*],)*
            })*
        ) => ($(
            #[test]
            fn $name() {
                unsafe {
                    $(
                        let v1 = [$($vec1)*];
                        let v2 = [$($vec2)*];
                        let v1_v128: v128 = mem::transmute(v1);
                        let v2_v128: v128 = mem::transmute(v2);
                        let v3_v128 = super::$f(v1_v128, v2_v128);
                        let mut v3 = [$($vec1)*];
                        drop(v3);
                        v3 = mem::transmute(v3_v128);

                        for (i, actual) in v3.iter().enumerate() {
                            let expected = (Wrapping(v1[i]) $op Wrapping(v2[i])).0;
                            assert_eq!(*actual, expected);
                        }
                    )*
                }
            }
        )*)
    }

    macro_rules! test_unop {
        (
            $($name:ident => {
                $(($op:tt | $f:ident) [$($vec1:tt)*],)*
            })*
        ) => ($(
            #[test]
            fn $name() {
                unsafe {
                    $(
                        let v1 = [$($vec1)*];
                        let v1_v128: v128 = mem::transmute(v1);
                        let v2_v128 = super::$f(v1_v128);
                        let mut v2 = [$($vec1)*];
                        drop(v2);
                        v2 = mem::transmute(v2_v128);

                        for (i, actual) in v2.iter().enumerate() {
                            let expected = ($op Wrapping(v1[i])).0;
                            assert_eq!(*actual, expected);
                        }
                    )*
                }
            }
        )*)
    }

    test_binop! {
        test_i8x16_add => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (+ | i8x16_add)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (+ | i8x16_add)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (+ | i8x16_add)
            [127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 9, -24],
        }
        test_i8x16_sub => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (- | i8x16_sub)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (- | i8x16_sub)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (- | i8x16_sub)
            [-127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 4, 8],
        }

        test_i16x8_add => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (+ | i16x8_add)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (+ | i16x8_add)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_sub => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (- | i16x8_sub)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (- | i16x8_sub)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_mul => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (* | i16x8_mul)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (* | i16x8_mul)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i32x4_add => {
            [0i32, 0, 0, 0] (+ | i32x4_add) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (+ | i32x4_add)
            [i32::MAX; 4],
        }

        test_i32x4_sub => {
            [0i32, 0, 0, 0] (- | i32x4_sub) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (- | i32x4_sub)
            [i32::MAX; 4],
        }

        test_i32x4_mul => {
            [0i32, 0, 0, 0] (* | i32x4_mul) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (* | i32x4_mul)
            [i32::MAX; 4],
        }

        // TODO: test_i64x2_add
        // TODO: test_i64x2_sub
    }

    test_unop! {
        test_i8x16_neg => {
            (- | i8x16_neg)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            (- | i8x16_neg)
            [-2i8, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            (- | i8x16_neg)
            [-127i8, -44, 43, 126, 4, -128, 127, -59, -43, 39, -69, 79, -3, 35, 83, 13],
        }

        test_i16x8_neg => {
            (- | i16x8_neg) [1i16, 1, 1, 1, 1, 1, 1, 1],
            (- | i16x8_neg) [2i16, 0x7fff, !0, 4, 42, -5, 33, -4847],
        }

        test_i32x4_neg => {
            (- | i32x4_neg) [1i32, 2, 3, 4],
            (- | i32x4_neg) [i32::MIN, i32::MAX, 0, 4],
        }

        // TODO: test_i64x2_neg
    }

    macro_rules! floating_point {
        (f32) => {
            true
        };
        (f64) => {
            true
        };
        ($id:ident) => {
            false
        };
    }

    trait IsNan: Sized {
        fn is_nan(self) -> bool {
            false
        }
    }
    impl IsNan for i8 {}
    impl IsNan for i16 {}
    impl IsNan for i32 {}
    impl IsNan for i64 {}

    macro_rules! test_bop {
         ($id:ident[$ety:ident; $ecount:expr] |
          $binary_op:ident [$op_test_id:ident] :
          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
             test_bop!(
                 $id[$ety; $ecount] => $ety | $binary_op [ $op_test_id ]:
                 ([$($in_a),*], [$($in_b),*]) => [$($out),*]
             );

         };
         ($id:ident[$ety:ident; $ecount:expr] => $oty:ident |
          $binary_op:ident [$op_test_id:ident] :
          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let b_input: [$ety; $ecount] = [$($in_b),*];
                     let output: [$oty; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let b_vec_in: v128 = transmute(b_input);
                     let vec_res: v128 = $binary_op(a_vec_in, b_vec_in);

                     let res: [$oty; $ecount] = transmute(vec_res);

                     if !floating_point!($ety) {
                         assert_eq!(res, output);
                     } else {
                         for i in 0..$ecount {
                             let r = res[i];
                             let o = output[i];
                             assert_eq!(r.is_nan(), o.is_nan());
                             if !r.is_nan() {
                                 assert_eq!(r, o);
                             }
                         }
                     }
                 }
             }
         }
     }

    macro_rules! test_bops {
         ($id:ident[$ety:ident; $ecount:expr] |
          $binary_op:ident [$op_test_id:ident]:
          ([$($in_a:expr),*], $in_b:expr) => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let output: [$ety; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let vec_res: v128 = $binary_op(a_vec_in, $in_b);

                     let res: [$ety; $ecount] = transmute(vec_res);
                     assert_eq!(res, output);
                 }
             }
         }
     }

    macro_rules! test_uop {
         ($id:ident[$ety:ident; $ecount:expr] |
          $unary_op:ident [$op_test_id:ident]: [$($in_a:expr),*] => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let output: [$ety; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let vec_res: v128 = $unary_op(a_vec_in);

                     let res: [$ety; $ecount] = transmute(vec_res);
                     assert_eq!(res, output);
                 }
             }
         }
     }

    test_bops!(i8x16[i8; 16] | i8x16_shl[i8x16_shl_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -2, 4, 6, 8, 10, 12, -2, 2, 2, 2, 2, 2, 2, 2, 2]);
    test_bops!(i16x8[i16; 8] | i16x8_shl[i16x8_shl_test]:
                ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
                [0, -2, 4, 6, 8, 10, 12, -2]);
    test_bops!(i32x4[i32; 4] | i32x4_shl[i32x4_shl_test]:
                ([0, -1, 2, 3], 1) => [0, -2, 4, 6]);
    test_bops!(i64x2[i64; 2] | i64x2_shl[i64x2_shl_test]:
                ([0, -1], 1) => [0, -2]);

    test_bops!(i8x16[i8; 16] | i8x16_shr_s[i8x16_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -1, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | i16x8_shr_s[i16x8_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
               [0, -1, 1, 1, 2, 2, 3, i16::MAX / 2]);
    test_bops!(i32x4[i32; 4] | i32x4_shr_s[i32x4_shr_s_test]:
               ([0, -1, 2, 3], 1) => [0, -1, 1, 1]);
    test_bops!(i64x2[i64; 2] | i64x2_shr_s[i64x2_shr_s_test]:
               ([0, -1], 1) => [0, -1]);

    test_bops!(i8x16[i8; 16] | i8x16_shr_u[i8x16_uhr_u_test]:
                ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
                [0, i8::MAX, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | i16x8_shr_u[i16x8_uhr_u_test]:
                ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
                [0, i16::MAX, 1, 1, 2, 2, 3, i16::MAX / 2]);
    test_bops!(i32x4[i32; 4] | i32x4_shr_u[i32x4_uhr_u_test]:
                ([0, -1, 2, 3], 1) => [0, i32::MAX, 1, 1]);
    test_bops!(i64x2[i64; 2] | i64x2_shr_u[i64x2_uhr_u_test]:
                ([0, -1], 1) => [0, i64::MAX]);

    #[test]
    fn v128_bitwise_logical_ops() {
        unsafe {
            let a: [u32; 4] = [u32::MAX, 0, u32::MAX, 0];
            let b: [u32; 4] = [u32::MAX; 4];
            let c: [u32; 4] = [0; 4];

            let vec_a: v128 = transmute(a);
            let vec_b: v128 = transmute(b);
            let vec_c: v128 = transmute(c);

            let r: v128 = v128_and(vec_a, vec_a);
            compare_bytes(r, vec_a);
            let r: v128 = v128_and(vec_a, vec_b);
            compare_bytes(r, vec_a);
            let r: v128 = v128_or(vec_a, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128_not(vec_b);
            compare_bytes(r, vec_c);
            let r: v128 = v128_xor(vec_a, vec_c);
            compare_bytes(r, vec_a);

            let r: v128 = v128_bitselect(vec_b, vec_c, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128_bitselect(vec_b, vec_c, vec_c);
            compare_bytes(r, vec_c);
            let r: v128 = v128_bitselect(vec_b, vec_c, vec_a);
            compare_bytes(r, vec_a);
        }
    }

    macro_rules! test_bool_red {
         ([$test_id:ident, $any:ident, $all:ident] | [$($true:expr),*] | [$($false:expr),*] | [$($alt:expr),*]) => {
             #[test]
             fn $test_id() {
                 unsafe {
                     let vec_a: v128 = transmute([$($true),*]); // true
                     let vec_b: v128 = transmute([$($false),*]); // false
                     let vec_c: v128 = transmute([$($alt),*]); // alternating

                     // TODO
                     // assert_eq!($any(vec_a), true);
                     // assert_eq!($any(vec_b), false);
                     // assert_eq!($any(vec_c), true);

                     assert_eq!($all(vec_a), true);
                     assert_eq!($all(vec_b), false);
                     assert_eq!($all(vec_c), false);
                 }
             }
         }
     }

    test_bool_red!(
        [i8x16_boolean_reductions, v128_any_true, i8x16_all_true]
            | [1_i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            | [0_i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            | [1_i8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        [i16x8_boolean_reductions, v128_any_true, i16x8_all_true]
            | [1_i16, 1, 1, 1, 1, 1, 1, 1]
            | [0_i16, 0, 0, 0, 0, 0, 0, 0]
            | [1_i16, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        [i32x4_boolean_reductions, v128_any_true, i32x4_all_true]
            | [1_i32, 1, 1, 1]
            | [0_i32, 0, 0, 0]
            | [1_i32, 0, 1, 0]
    );

    test_bop!(i8x16[i8; 16] | i8x16_eq[i8x16_eq_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_eq[i16x8_eq_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_eq[i32x4_eq_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_eq[f32x4_eq_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_eq[f64x2_eq_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | i8x16_ne[i8x16_ne_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_ne[i16x8_ne_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_ne[i32x4_ne_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_ne[f32x4_ne_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_ne[f64x2_ne_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_lt_s[i8x16_lt_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_lt_s[i16x8_lt_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_lt_s[i32x4_lt_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_lt[f32x4_lt_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_lt[f64x2_lt_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_gt_s[i8x16_gt_test]:
           ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_gt_s[i16x8_gt_test]:
               ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_gt_s[i32x4_gt_test]:
               ([0, 2, 2, 4], [0, 1, 2, 3]) => [0, -1, 0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_gt[f32x4_gt_test]:
               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_gt[f64x2_gt_test]: ([0., 2.], [0., 1.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_ge_s[i8x16_ge_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_ge_s[i16x8_ge_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_ge_s[i32x4_ge_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_ge[f32x4_ge_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_ge[f64x2_ge_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | i8x16_le_s[i8x16_le_test]:
               ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
               ) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_le_s[i16x8_le_test]:
               ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_le_s[i32x4_le_test]:
               ([0, 2, 2, 4], [0, 1, 2, 3]) => [-1, 0, -1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_le[f32x4_le_test]:
               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [-1, 0, -1, -0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_le[f64x2_le_test]: ([0., 2.], [0., 1.]) => [-1, 0]);

    test_uop!(f32x4[f32; 4] | f32x4_neg[f32x4_neg_test]: [0., 1., 2., 3.] => [ 0., -1., -2., -3.]);
    test_uop!(f32x4[f32; 4] | f32x4_abs[f32x4_abs_test]: [0., -1., 2., -3.] => [ 0., 1., 2., 3.]);
    test_bop!(f32x4[f32; 4] | f32x4_min[f32x4_min_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., -3., -4., 8.]);
    test_bop!(f32x4[f32; 4] | f32x4_min[f32x4_min_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
              => [0., -3., -4., std::f32::NAN]);
    test_bop!(f32x4[f32; 4] | f32x4_max[f32x4_max_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -1., 7., 10.]);
    test_bop!(f32x4[f32; 4] | f32x4_max[f32x4_max_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
              => [1., -1., 7., std::f32::NAN]);
    test_bop!(f32x4[f32; 4] | f32x4_add[f32x4_add_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -4., 3., 18.]);
    test_bop!(f32x4[f32; 4] | f32x4_sub[f32x4_sub_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [-1., 2., 11., -2.]);
    test_bop!(f32x4[f32; 4] | f32x4_mul[f32x4_mul_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., 3., -28., 80.]);
    test_bop!(f32x4[f32; 4] | f32x4_div[f32x4_div_test]:
              ([0., -8., 70., 8.], [1., 4., 10., 2.]) => [0., -2., 7., 4.]);

    test_uop!(f64x2[f64; 2] | f64x2_neg[f64x2_neg_test]: [0., 1.] => [ 0., -1.]);
    test_uop!(f64x2[f64; 2] | f64x2_abs[f64x2_abs_test]: [0., -1.] => [ 0., 1.]);
    test_bop!(f64x2[f64; 2] | f64x2_min[f64x2_min_test]:
               ([0., -1.], [1., -3.]) => [0., -3.]);
    test_bop!(f64x2[f64; 2] | f64x2_min[f64x2_min_test_nan]:
               ([7., 8.], [-4., std::f64::NAN])
               => [ -4., std::f64::NAN]);
    test_bop!(f64x2[f64; 2] | f64x2_max[f64x2_max_test]:
               ([0., -1.], [1., -3.]) => [1., -1.]);
    test_bop!(f64x2[f64; 2] | f64x2_max[f64x2_max_test_nan]:
               ([7., 8.], [ -4., std::f64::NAN])
               => [7., std::f64::NAN]);
    test_bop!(f64x2[f64; 2] | f64x2_add[f64x2_add_test]:
               ([0., -1.], [1., -3.]) => [1., -4.]);
    test_bop!(f64x2[f64; 2] | f64x2_sub[f64x2_sub_test]:
               ([0., -1.], [1., -3.]) => [-1., 2.]);
    test_bop!(f64x2[f64; 2] | f64x2_mul[f64x2_mul_test]:
               ([0., -1.], [1., -3.]) => [0., 3.]);
    test_bop!(f64x2[f64; 2] | f64x2_div[f64x2_div_test]:
               ([0., -8.], [1., 4.]) => [0., -2.]);

    macro_rules! test_conv {
        ($test_id:ident | $conv_id:ident | $to_ty:ident | $from:expr,  $to:expr) => {
            #[test]
            fn $test_id() {
                unsafe {
                    let from: v128 = transmute($from);
                    let to: v128 = transmute($to);

                    let r: v128 = $conv_id(from);

                    compare_bytes(r, to);
                }
            }
        };
    }

    test_conv!(
        f32x4_convert_s_i32x4 | f32x4_convert_i32x4_s | f32x4 | [1_i32, 2, 3, 4],
        [1_f32, 2., 3., 4.]
    );
    test_conv!(
        f32x4_convert_u_i32x4 | f32x4_convert_i32x4_u | f32x4 | [u32::MAX, 2, 3, 4],
        [u32::MAX as f32, 2., 3., 4.]
    );

    // FIXME: this fails, and produces 0 instead of saturating at i32::MAX
    // test_conv!(
    //     i32x4_trunc_s_f32x4_sat
    //         | i32x4_trunc_sat_f32x4_s
    //         | i32x4
    //         | [f32::NAN, 2., (i32::MAX as f32 + 1.), 4.],
    //     [0, 2, i32::MAX, 4]
    // );
    // FIXME: add other saturating tests
}
