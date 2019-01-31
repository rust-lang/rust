//! This module implements the [WebAssembly `SIMD128` ISA].
//!
//! [WebAssembly `SIMD128` ISA]:
//! https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md

#![allow(non_camel_case_types)]

use core_arch::simd::*;
use core_arch::simd_llvm::*;
use marker::Sized;
use mem;
use ptr;
use u8;

#[cfg(test)]
use stdsimd_test::assert_instr;
#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test;

types! {
    /// WASM-specific 128-bit wide SIMD vector type.
    // N.B., internals here are arbitrary.
    pub struct v128(i32, i32, i32, i32);
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdimd_internal", issue = "0")]
pub(crate) trait v128Ext: Sized {
    fn as_v128(self) -> v128;

    #[inline]
    fn as_u8x16(self) -> u8x16 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_u16x8(self) -> u16x8 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_u32x4(self) -> u32x4 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_u64x2(self) -> u64x2 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_i8x16(self) -> i8x16 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_i16x8(self) -> i16x8 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_i32x4(self) -> i32x4 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_i64x2(self) -> i64x2 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_f32x4(self) -> f32x4 {
        unsafe { mem::transmute(self.as_v128()) }
    }

    #[inline]
    fn as_f64x2(self) -> f64x2 {
        unsafe { mem::transmute(self.as_v128()) }
    }
}

impl v128Ext for v128 {
    #[inline]
    fn as_v128(self) -> Self {
        self
    }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.wasm.anytrue.v16i8"]
    fn llvm_i8x16_any_true(x: i8x16) -> i32;
    #[link_name = "llvm.wasm.alltrue.v16i8"]
    fn llvm_i8x16_all_true(x: i8x16) -> i32;
    #[link_name = "llvm.sadd.sat.v16i8"]
    fn llvm_i8x16_add_saturate_s(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.uadd.sat.v16i8"]
    fn llvm_i8x16_add_saturate_u(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.sub.saturate.signed.v16i8"]
    fn llvm_i8x16_sub_saturate_s(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.wasm.sub.saturate.unsigned.v16i8"]
    fn llvm_i8x16_sub_saturate_u(a: i8x16, b: i8x16) -> i8x16;

    #[link_name = "llvm.wasm.anytrue.v8i16"]
    fn llvm_i16x8_any_true(x: i16x8) -> i32;
    #[link_name = "llvm.wasm.alltrue.v8i16"]
    fn llvm_i16x8_all_true(x: i16x8) -> i32;
    #[link_name = "llvm.sadd.sat.v8i16"]
    fn llvm_i16x8_add_saturate_s(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.uadd.sat.v8i16"]
    fn llvm_i16x8_add_saturate_u(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.sub.saturate.signed.v8i16"]
    fn llvm_i16x8_sub_saturate_s(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.wasm.sub.saturate.unsigned.v8i16"]
    fn llvm_i16x8_sub_saturate_u(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.wasm.anytrue.v4i32"]
    fn llvm_i32x4_any_true(x: i32x4) -> i32;
    #[link_name = "llvm.wasm.alltrue.v4i32"]
    fn llvm_i32x4_all_true(x: i32x4) -> i32;

    #[link_name = "llvm.wasm.anytrue.v2i64"]
    fn llvm_i64x2_any_true(x: i64x2) -> i32;
    #[link_name = "llvm.wasm.alltrue.v2i64"]
    fn llvm_i64x2_all_true(x: i64x2) -> i32;

    #[link_name = "llvm.fabs.v4f32"]
    fn llvm_f32x4_abs(x: f32x4) -> f32x4;
    #[link_name = "llvm.sqrt.v4f32"]
    fn llvm_f32x4_sqrt(x: f32x4) -> f32x4;
    #[link_name = "llvm.minimum.v4f32"]
    fn llvm_f32x4_min(x: f32x4, y: f32x4) -> f32x4;
    #[link_name = "llvm.maximum.v4f32"]
    fn llvm_f32x4_max(x: f32x4, y: f32x4) -> f32x4;
    #[link_name = "llvm.fabs.v2f64"]
    fn llvm_f64x2_abs(x: f64x2) -> f64x2;
    #[link_name = "llvm.sqrt.v2f64"]
    fn llvm_f64x2_sqrt(x: f64x2) -> f64x2;
    #[link_name = "llvm.minimum.v2f64"]
    fn llvm_f64x2_min(x: f64x2, y: f64x2) -> f64x2;
    #[link_name = "llvm.maximum.v2f64"]
    fn llvm_f64x2_max(x: f64x2, y: f64x2) -> f64x2;

    #[link_name = "llvm.wasm.bitselect.v16i8"]
    fn llvm_bitselect(a: i8x16, b: i8x16, c: i8x16) -> i8x16;
}

/// Loads a `v128` vector from the given heap address.
#[inline]
#[cfg_attr(test, assert_instr(v128.load))]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    ptr::read(m)
}

/// Stores a `v128` vector to the given heap address.
#[inline]
#[cfg_attr(test, assert_instr(v128.store))]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    ptr::write(m, a)
}

/// Materializes a constant SIMD value from the immediate operands.
///
/// The `v128.const` instruction is encoded with 16 immediate bytes
/// `imm` which provide the bits of the vector directly.
#[inline]
#[rustc_args_required_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
#[cfg_attr(test, assert_instr(
    v128.const,
    a0 = 0,
    a1 = 1,
    a2 = 2,
    a3 = 3,
    a4 = 4,
    a5 = 5,
    a6 = 6,
    a7 = 7,
    a8 = 8,
    a9 = 9,
    a10 = 10,
    a11 = 11,
    a12 = 12,
    a13 = 13,
    a14 = 14,
    a15 = 15,
))]
pub const fn v128_const(
    a0: u8,
    a1: u8,
    a2: u8,
    a3: u8,
    a4: u8,
    a5: u8,
    a6: u8,
    a7: u8,
    a8: u8,
    a9: u8,
    a10: u8,
    a11: u8,
    a12: u8,
    a13: u8,
    a14: u8,
    a15: u8,
) -> v128 {
    union U {
        imm: [u8; 16],
        vec: v128,
    }
    unsafe {
        U {
            imm: [
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
            ],
        }
        .vec
    }
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 16 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn i8x16_splat(a: i8) -> v128 {
    unsafe { mem::transmute(i8x16::splat(a)) }
}

/// Extracts a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 16.
#[inline]
#[rustc_args_required_const(1)]
pub unsafe fn i8x16_extract_lane(a: v128, imm: usize) -> i8 {
    #[cfg(test)]
    #[assert_instr(i8x16.extract_lane_s)]
    fn extract_lane_s(a: v128) -> i32 {
        unsafe { i8x16_extract_lane(a, 0) as i32 }
    }
    #[cfg(test)]
    #[assert_instr(i8x16.extract_lane_u)]
    fn extract_lane_u(a: v128) -> u32 {
        unsafe { i8x16_extract_lane(a, 0) as u32 }
    }
    simd_extract(a.as_i8x16(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 16.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i8x16_replace_lane(a: v128, imm: usize, val: i8) -> v128 {
    mem::transmute(simd_insert(a.as_i8x16(), imm as u32, val))
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 8 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn i16x8_splat(a: i16) -> v128 {
    unsafe { mem::transmute(i16x8::splat(a)) }
}

/// Extracts a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Extracts a the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 8.
#[inline]
#[rustc_args_required_const(1)]
pub unsafe fn i16x8_extract_lane(a: v128, imm: usize) -> i16 {
    #[cfg(test)]
    #[assert_instr(i16x8.extract_lane_s)]
    fn extract_lane_s(a: v128) -> i32 {
        unsafe { i16x8_extract_lane(a, 0) as i32 }
    }
    #[cfg(test)]
    #[assert_instr(i16x8.extract_lane_u)]
    fn extract_lane_u(a: v128) -> u32 {
        unsafe { i16x8_extract_lane(a, 0) as u32 }
    }
    simd_extract(a.as_i16x8(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 8.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i16x8_replace_lane(a: v128, imm: usize, val: i16) -> v128 {
    mem::transmute(simd_insert(a.as_i16x8(), imm as u32, val))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn i32x4_splat(a: i32) -> v128 {
    unsafe { mem::transmute(i32x4::splat(a)) }
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 4.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extract_lane_s, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i32x4_extract_lane(a: v128, imm: usize) -> i32 {
    simd_extract(a.as_i32x4(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 4.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i32x4_replace_lane(a: v128, imm: usize, val: i32) -> v128 {
    mem::transmute(simd_insert(a.as_i32x4(), imm as u32, val))
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn i64x2_splat(a: i64) -> v128 {
    unsafe { mem::transmute(i64x2::splat(a)) }
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 2.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extract_lane_s, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i64x2_extract_lane(a: v128, imm: usize) -> i64 {
    simd_extract(a.as_i64x2(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 2.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn i64x2_replace_lane(a: v128, imm: usize, val: i64) -> v128 {
    mem::transmute(simd_insert(a.as_i64x2(), imm as u32, val))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn f32x4_splat(a: f32) -> v128 {
    unsafe { mem::transmute(f32x4::splat(a)) }
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 4.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.extract_lane_s, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn f32x4_extract_lane(a: v128, imm: usize) -> f32 {
    simd_extract(a.as_f32x4(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 4.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn f32x4_replace_lane(a: v128, imm: usize, val: f32) -> v128 {
    mem::transmute(simd_insert(a.as_f32x4(), imm as u32, val))
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
pub fn f64x2_splat(a: f64) -> v128 {
    unsafe { mem::transmute(f64x2::splat(a)) }
}

/// Extracts lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `imm` from `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 2.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.extract_lane_s, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn f64x2_extract_lane(a: v128, imm: usize) -> f64 {
    simd_extract(a.as_f64x2(), imm as u32)
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `imm` with `a`.
///
/// # Unsafety
///
/// This function has undefined behavior if `imm` is greater than or equal to
/// 2.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.replace_lane, imm = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn f64x2_replace_lane(a: v128, imm: usize, val: f64) -> v128 {
    mem::transmute(simd_insert(a.as_f64x2(), imm as u32, val))
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.eq))]
pub fn i8x16_eq(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_eq::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ne))]
pub fn i8x16_ne(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ne::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_s))]
pub fn i8x16_lt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_u))]
pub fn i8x16_lt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i8x16>(a.as_u8x16(), b.as_u8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_s))]
pub fn i8x16_gt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_u))]
pub fn i8x16_gt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i8x16>(a.as_u8x16(), b.as_u8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_s))]
pub fn i8x16_le_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_u))]
pub fn i8x16_le_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i8x16>(a.as_u8x16(), b.as_u8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_s))]
pub fn i8x16_ge_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i8x16>(a.as_i8x16(), b.as_i8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_u))]
pub fn i8x16_ge_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i8x16>(a.as_u8x16(), b.as_u8x16())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.eq))]
pub fn i16x8_eq(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_eq::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ne))]
pub fn i16x8_ne(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ne::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_s))]
pub fn i16x8_lt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_u))]
pub fn i16x8_lt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i16x8>(a.as_u16x8(), b.as_u16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_s))]
pub fn i16x8_gt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_u))]
pub fn i16x8_gt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i16x8>(a.as_u16x8(), b.as_u16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_s))]
pub fn i16x8_le_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_u))]
pub fn i16x8_le_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i16x8>(a.as_u16x8(), b.as_u16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_s))]
pub fn i16x8_ge_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i16x8>(a.as_i16x8(), b.as_i16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_u))]
pub fn i16x8_ge_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i16x8>(a.as_u16x8(), b.as_u16x8())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.eq))]
pub fn i32x4_eq(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_eq::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ne))]
pub fn i32x4_ne(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ne::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_s))]
pub fn i32x4_lt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_u))]
pub fn i32x4_lt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i32x4>(a.as_u32x4(), b.as_u32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_s))]
pub fn i32x4_gt_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_u))]
pub fn i32x4_gt_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i32x4>(a.as_u32x4(), b.as_u32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_s))]
pub fn i32x4_le_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_u))]
pub fn i32x4_le_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i32x4>(a.as_u32x4(), b.as_u32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_s))]
pub fn i32x4_ge_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i32x4>(a.as_i32x4(), b.as_i32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_u))]
pub fn i32x4_ge_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i32x4>(a.as_u32x4(), b.as_u32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.eq))]
pub fn f32x4_eq(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_eq::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ne))]
pub fn f32x4_ne(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ne::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.lt))]
pub fn f32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.gt))]
pub fn f32x4_gt(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.le))]
pub fn f32x4_le(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ge))]
pub fn f32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i32x4>(a.as_f32x4(), b.as_f32x4())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were equal, or all zeros if the elements were not equal.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.eq))]
pub fn f64x2_eq(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_eq::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise elements
/// were not equal, or all zeros if the elements were equal.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ne))]
pub fn f64x2_ne(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ne::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.lt))]
pub fn f64x2_lt(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_lt::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.gt))]
pub fn f64x2_gt(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_gt::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is less than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.le))]
pub fn f64x2_le(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_le::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the pairwise left
/// element is greater than the pairwise right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ge))]
pub fn f64x2_ge(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_ge::<_, i64x2>(a.as_f64x2(), b.as_f64x2())) }
}

/// Flips each bit of the 128-bit input vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.not))]
pub fn v128_not(a: v128) -> v128 {
    unsafe { mem::transmute(simd_xor(a.as_i64x2(), i64x2(!0, !0))) }
}

/// Performs a bitwise and of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.and))]
pub fn v128_and(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_and(a.as_i64x2(), b.as_i64x2())) }
}

/// Performs a bitwise or of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.or))]
pub fn v128_or(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_or(a.as_i64x2(), b.as_i64x2())) }
}

/// Performs a bitwise xor of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.xor))]
pub fn v128_xor(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_xor(a.as_i64x2(), b.as_i64x2())) }
}

/// Use the bitmask in `c` to select bits from `v1` when 1 and `v2` when 0.
#[inline]
#[cfg_attr(test, assert_instr(v128.bitselect))]
pub fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    unsafe { mem::transmute(llvm_bitselect(c.as_i8x16(), v1.as_i8x16(), v2.as_i8x16())) }
}

/// Negates a 128-bit vectors intepreted as sixteen 8-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i8x16.neg))]
pub fn i8x16_neg(a: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i8x16(), i8x16::splat(-1))) }
}

/// Returns 1 if any lane is nonzero or 0 if all lanes are zero.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.any_true))]
pub fn i8x16_any_true(a: v128) -> i32 {
    unsafe { llvm_i8x16_any_true(a.as_i8x16()) }
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.all_true))]
pub fn i8x16_all_true(a: v128) -> i32 {
    unsafe { llvm_i8x16_all_true(a.as_i8x16()) }
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shl))]
pub fn i8x16_shl(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shl(a.as_i8x16(), i8x16::splat(amt as i8))) }
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shl))]
pub fn i8x16_shr_s(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_i8x16(), i8x16::splat(amt as i8))) }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shl))]
pub fn i8x16_shr_u(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_u8x16(), u8x16::splat(amt as u8))) }
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add))]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_i8x16(), b.as_i8x16())) }
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit signed
/// integers, saturating on overflow to `i8::max_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_saturate_s))]
pub fn i8x16_add_saturate_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i8x16_add_saturate_s(a.as_i8x16(), b.as_i8x16())) }
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit unsigned
/// integers, saturating on overflow to `u8::max_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_saturate_u))]
pub fn i8x16_add_saturate_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i8x16_add_saturate_u(a.as_i8x16(), b.as_i8x16())) }
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub))]
pub fn i8x16_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_i8x16(), b.as_i8x16())) }
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// signed integers, saturating on overflow to `i8::min_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_saturate_s))]
pub fn i8x16_sub_saturate_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i8x16_sub_saturate_s(a.as_i8x16(), b.as_i8x16())) }
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_saturate_u))]
pub fn i8x16_sub_saturate_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i8x16_sub_saturate_u(a.as_i8x16(), b.as_i8x16())) }
}

/// Multiplies two 128-bit vectors as if they were two packed sixteen 8-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.mul))]
pub fn i8x16_mul(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i8x16(), b.as_i8x16())) }
}

/// Negates a 128-bit vectors intepreted as eight 16-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i16x8.neg))]
pub fn i16x8_neg(a: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i16x8(), i16x8::splat(-1))) }
}

/// Returns 1 if any lane is nonzero or 0 if all lanes are zero.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.any_true))]
pub fn i16x8_any_true(a: v128) -> i32 {
    unsafe { llvm_i16x8_any_true(a.as_i16x8()) }
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.all_true))]
pub fn i16x8_all_true(a: v128) -> i32 {
    unsafe { llvm_i16x8_all_true(a.as_i16x8()) }
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shl))]
pub fn i16x8_shl(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shl(a.as_i16x8(), i16x8::splat(amt as i16))) }
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shl))]
pub fn i16x8_shr_s(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_i16x8(), i16x8::splat(amt as i16))) }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shl))]
pub fn i16x8_shr_u(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_u16x8(), u16x8::splat(amt as u16))) }
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add))]
pub fn i16x8_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_i16x8(), b.as_i16x8())) }
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit signed
/// integers, saturating on overflow to `i16::max_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_saturate_s))]
pub fn i16x8_add_saturate_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i16x8_add_saturate_s(a.as_i16x8(), b.as_i16x8())) }
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit unsigned
/// integers, saturating on overflow to `u16::max_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_saturate_u))]
pub fn i16x8_add_saturate_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i16x8_add_saturate_u(a.as_i16x8(), b.as_i16x8())) }
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub))]
pub fn i16x8_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_i16x8(), b.as_i16x8())) }
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers, saturating on overflow to `i16::min_value()`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_saturate_s))]
pub fn i16x8_sub_saturate_s(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i16x8_sub_saturate_s(a.as_i16x8(), b.as_i16x8())) }
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_saturate_u))]
pub fn i16x8_sub_saturate_u(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_i16x8_sub_saturate_u(a.as_i16x8(), b.as_i16x8())) }
}

/// Multiplies two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.mul))]
pub fn i16x8_mul(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i16x8(), b.as_i16x8())) }
}

/// Negates a 128-bit vectors intepreted as four 32-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i32x4.neg))]
pub fn i32x4_neg(a: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i32x4(), i32x4::splat(-1))) }
}

/// Returns 1 if any lane is nonzero or 0 if all lanes are zero.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.any_true))]
pub fn i32x4_any_true(a: v128) -> i32 {
    unsafe { llvm_i32x4_any_true(a.as_i32x4()) }
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.all_true))]
pub fn i32x4_all_true(a: v128) -> i32 {
    unsafe { llvm_i32x4_all_true(a.as_i32x4()) }
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shl))]
pub fn i32x4_shl(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shl(a.as_i32x4(), i32x4::splat(amt as i32))) }
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shl))]
pub fn i32x4_shr_s(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_i32x4(), i32x4::splat(amt as i32))) }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shl))]
pub fn i32x4_shr_u(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_u32x4(), u32x4::splat(amt as u32))) }
}

/// Adds two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.add))]
pub fn i32x4_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_i32x4(), b.as_i32x4())) }
}

/// Subtracts two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.sub))]
pub fn i32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_i32x4(), b.as_i32x4())) }
}

/// Multiplies two 128-bit vectors as if they were two packed four 32-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.mul))]
pub fn i32x4_mul(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i32x4(), b.as_i32x4())) }
}

/// Negates a 128-bit vectors intepreted as two 64-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i32x4.neg))]
pub fn i64x2_neg(a: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_i64x2(), i64x2::splat(-1))) }
}

/// Returns 1 if any lane is nonzero or 0 if all lanes are zero.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.any_true))]
pub fn i64x2_any_true(a: v128) -> i32 {
    unsafe { llvm_i64x2_any_true(a.as_i64x2()) }
}

/// Returns 1 if all lanes are nonzero or 0 if any lane is nonzero.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.all_true))]
pub fn i64x2_all_true(a: v128) -> i32 {
    unsafe { llvm_i64x2_all_true(a.as_i64x2()) }
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shl))]
pub fn i64x2_shl(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shl(a.as_i64x2(), i64x2::splat(amt as i64))) }
}

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shl))]
pub fn i64x2_shr_s(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_i64x2(), i64x2::splat(amt as i64))) }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shl))]
pub fn i64x2_shr_u(a: v128, amt: u32) -> v128 {
    unsafe { mem::transmute(simd_shr(a.as_u64x2(), u64x2::splat(amt as u64))) }
}

/// Adds two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.add))]
pub fn i64x2_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_i64x2(), b.as_i64x2())) }
}

/// Subtracts two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.sub))]
pub fn i64x2_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_i64x2(), b.as_i64x2())) }
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.abs))]
pub fn f32x4_abs(a: v128) -> v128 {
    unsafe { mem::transmute(llvm_f32x4_abs(a.as_f32x4())) }
}

/// Negates each lane of a 128-bit vector interpreted as four 32-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.abs))]
pub fn f32x4_neg(a: v128) -> v128 {
    unsafe { f32x4_mul(a, mem::transmute(f32x4(-1.0, -1.0, -1.0, -1.0))) }
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sqrt))]
pub fn f32x4_sqrt(a: v128) -> v128 {
    unsafe { mem::transmute(llvm_f32x4_sqrt(a.as_f32x4())) }
}

/// Adds pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.add))]
pub fn f32x4_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_f32x4(), b.as_f32x4())) }
}

/// Subtracts pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sub))]
pub fn f32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_f32x4(), b.as_f32x4())) }
}

/// Multiplies pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.mul))]
pub fn f32x4_mul(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_f32x4(), b.as_f32x4())) }
}

/// Divides pairwise lanes of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.div))]
pub fn f32x4_div(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_div(a.as_f32x4(), b.as_f32x4())) }
}

/// Calculates the minimum of pairwise lanes of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.min))]
pub fn f32x4_min(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_f32x4_min(a.as_f32x4(), b.as_f32x4())) }
}

/// Calculates the maximum of pairwise lanes of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.max))]
pub fn f32x4_max(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_f32x4_max(a.as_f32x4(), b.as_f32x4())) }
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.abs))]
pub fn f64x2_abs(a: v128) -> v128 {
    unsafe { mem::transmute(llvm_f64x2_abs(a.as_f64x2())) }
}

/// Negates each lane of a 128-bit vector interpreted as two 64-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.abs))]
pub fn f64x2_neg(a: v128) -> v128 {
    unsafe { f64x2_mul(a, mem::transmute(f64x2(-1.0, -1.0))) }
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sqrt))]
pub fn f64x2_sqrt(a: v128) -> v128 {
    unsafe { mem::transmute(llvm_f64x2_sqrt(a.as_f64x2())) }
}

/// Adds pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.add))]
pub fn f64x2_add(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_add(a.as_f64x2(), b.as_f64x2())) }
}

/// Subtracts pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sub))]
pub fn f64x2_sub(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_sub(a.as_f64x2(), b.as_f64x2())) }
}

/// Multiplies pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.mul))]
pub fn f64x2_mul(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_mul(a.as_f64x2(), b.as_f64x2())) }
}

/// Divides pairwise lanes of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.div))]
pub fn f64x2_div(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(simd_div(a.as_f64x2(), b.as_f64x2())) }
}

/// Calculates the minimum of pairwise lanes of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.min))]
pub fn f64x2_min(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_f64x2_min(a.as_f64x2(), b.as_f64x2())) }
}

/// Calculates the maximum of pairwise lanes of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.max))]
pub fn f64x2_max(a: v128, b: v128) -> v128 {
    unsafe { mem::transmute(llvm_f64x2_max(a.as_f64x2(), b.as_f64x2())) }
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit signed integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr("i32x4.trunc_s/f32x4:sat"))]
pub fn i32x4_trunc_s_f32x4_sat(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, i32x4>(a.as_f32x4())) }
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit unsigned integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr("i32x4.trunc_u/f32x4:sat"))]
pub fn i32x4_trunc_u_f32x4_sat(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, u32x4>(a.as_f32x4())) }
}

/// Converts a 128-bit vector interpreted as two 64-bit floating point numbers
/// into a 128-bit vector of two 64-bit signed integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr("i32x4.trunc_s/f32x4:sat"))]
pub fn i64x2_trunc_s_f64x2_sat(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, i64x2>(a.as_f64x2())) }
}

/// Converts a 128-bit vector interpreted as two 64-bit floating point numbers
/// into a 128-bit vector of two 64-bit unsigned integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr("i64x2.trunc_u/f64x2:sat"))]
pub fn i64x2_trunc_u_f64x2_sat(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, u64x2>(a.as_f64x2())) }
}

/// Converts a 128-bit vector interpreted as four 32-bit signed integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr("f32x4.convert_s/i32x4"))]
pub fn f32x4_convert_s_i32x4(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, f32x4>(a.as_i32x4())) }
}

/// Converts a 128-bit vector interpreted as four 32-bit unsigned integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr("f32x4.convert_u/i32x4"))]
pub fn f32x4_convert_u_i32x4(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, f32x4>(a.as_u32x4())) }
}

/// Converts a 128-bit vector interpreted as two 64-bit signed integers into a
/// 128-bit vector of two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr("f64x2.convert_s/i64x2"))]
pub fn f64x2_convert_s_i64x2(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, f64x2>(a.as_i64x2())) }
}

/// Converts a 128-bit vector interpreted as two 64-bit unsigned integers into a
/// 128-bit vector of two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr("f64x2.convert_u/i64x2"))]
pub fn f64x2_convert_u_i64x2(a: v128) -> v128 {
    unsafe { mem::transmute(simd_cast::<_, f64x2>(a.as_u64x2())) }
}

// #[cfg(test)]
// pub mod tests {
//     use super::*;
//     use std;
//     use std::mem;
//     use std::prelude::v1::*;
//     use wasm_bindgen_test::*;
//
//     fn compare_bytes(a: v128, b: v128) {
//         let a: [u8; 16] = unsafe { mem::transmute(a) };
//         let b: [u8; 16] = unsafe { mem::transmute(b) };
//         assert_eq!(a, b);
//     }
//
//     #[wasm_bindgen_test]
//     fn v128_const() {
//         const A: v128 = unsafe {
//             v128::const_([
//                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//             ])
//         };
//         compare_bytes(A, A);
//     }
//
//     macro_rules! test_splat {
//         ($test_id:ident: $id:ident($val:expr) => $($vals:expr),*) => {
//             #[wasm_bindgen_test]
//             fn $test_id() {
//                 const A: v128 = unsafe {
//                     $id::splat($val)
//                 };
//                 const B: v128 = unsafe {
//                     v128::const_([$($vals),*])
//                 };
//                 compare_bytes(A, B);
//             }
//         }
//     }
//
//     test_splat!(i8x16_splat: i8x16(42) => 42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42);
//     test_splat!(i16x8_splat: i16x8(42) => 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0);
//     test_splat!(i32x4_splat: i32x4(42) => 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0);
//     test_splat!(i64x2_splat: i64x2(42) => 42, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0);
//     test_splat!(f32x4_splat: f32x4(42.) => 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66);
//     test_splat!(f64x2_splat: f64x2(42.) => 0, 0, 0, 0, 0, 0, 69, 64, 0, 0, 0, 0, 0, 0, 69, 64);
//
//     // tests extract and replace lanes
//     macro_rules! test_extract {
//         ($test_id:ident: $id:ident[$ety:ident] => $extract_fn:ident | [$val:expr; $count:expr]
//          | [$($vals:expr),*] => ($other:expr)
//          | $($ids:expr),*) => {
//             #[wasm_bindgen_test]
//             fn $test_id() {
//                 unsafe {
//                     // splat vector and check that all indices contain the same value
//                     // splatted:
//                     const A: v128 = unsafe {
//                         $id::splat($val)
//                     };
//                     $(
//                         assert_eq!($id::$extract_fn(A, $ids) as $ety, $val);
//                     )*;
//
//                     // create a vector from array and check that the indices contain
//                     // the same values as in the array:
//                     let arr: [$ety; $count] = [$($vals),*];
//                     let mut vec: v128 = mem::transmute(arr);
//                     $(
//                         assert_eq!($id::$extract_fn(vec, $ids) as $ety, arr[$ids]);
//                     )*;
//
//                     // replace lane 0 with another value
//                     vec = $id::replace_lane(vec, 0, $other);
//                     assert_ne!($id::$extract_fn(vec, 0) as $ety, arr[0]);
//                     assert_eq!($id::$extract_fn(vec, 0) as $ety, $other);
//                 }
//             }
//         }
//     }
//
//     test_extract!(i8x16_extract_u: i8x16[u8] => extract_lane_u | [255; 16]
//                   | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] => (42)
//                   | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
//     );
//     test_extract!(i8x16_extract_s: i8x16[i8] => extract_lane_s | [-122; 16]
//                   | [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15] => (-42)
//                   | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
//     );
//
//     test_extract!(i16x8_extract_u: i16x8[u16] => extract_lane_u | [255; 8]
//                   | [0, 1, 2, 3, 4, 5, 6, 7]  => (42) | 0, 1, 2, 3, 4, 5, 6, 7
//     );
//     test_extract!(i16x8_extract_s: i16x8[i16] => extract_lane_s | [-122; 8]
//                   | [0, -1, 2, -3, 4, -5, 6, -7]  => (-42) | 0, 1, 2, 3, 4, 5, 6, 7
//     );
//     test_extract!(i32x4_extract: i32x4[i32] => extract_lane | [-122; 4]
//                   | [0, -1, 2, -3]  => (42) | 0, 1, 2, 3
//     );
//     test_extract!(i64x2_extract: i64x2[i64] => extract_lane | [-122; 2]
//                   | [0, -1]  => (42) | 0, 1
//     );
//     test_extract!(f32x4_extract: f32x4[f32] => extract_lane | [-122.; 4]
//                   | [0., -1., 2., -3.]  => (42.) | 0, 1, 2, 3
//     );
//     test_extract!(f64x2_extract: f64x2[f64] => extract_lane | [-122.; 2]
//                   | [0., -1.]  => (42.) | 0, 1
//     );
//
//     #[wasm_bindgen_test]
//     fn v8x16_shuffle() {
//         unsafe {
//             let a = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
//             let b = [
//                 16_u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
//                 31,
//             ];
//
//             let vec_a: v128 = mem::transmute(a);
//             let vec_b: v128 = mem::transmute(b);
//
//             let vec_r = v8x16_shuffle!(
//                 vec_a,
//                 vec_b,
//                 [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
//             );
//
//             let e =
//                 [0_u8, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
//             let vec_e: v128 = mem::transmute(e);
//             compare_bytes(vec_r, vec_e);
//         }
//     }
//
//     macro_rules! floating_point {
//         (f32) => {
//             true
//         };
//         (f64) => {
//             true
//         };
//         ($id:ident) => {
//             false
//         };
//     }
//
//     trait IsNan: Sized {
//         fn is_nan(self) -> bool {
//             false
//         }
//     }
//     impl IsNan for i8 {}
//     impl IsNan for i16 {}
//     impl IsNan for i32 {}
//     impl IsNan for i64 {}
//
//     macro_rules! test_bop {
//         ($id:ident[$ety:ident; $ecount:expr] |
//          $binary_op:ident [$op_test_id:ident] :
//          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
//             test_bop!(
//                 $id[$ety; $ecount] => $ety | $binary_op [ $op_test_id ]:
//                 ([$($in_a),*], [$($in_b),*]) => [$($out),*]
//             );
//
//         };
//         ($id:ident[$ety:ident; $ecount:expr] => $oty:ident |
//          $binary_op:ident [$op_test_id:ident] :
//          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
//             #[wasm_bindgen_test]
//             fn $op_test_id() {
//                 unsafe {
//                     let a_input: [$ety; $ecount] = [$($in_a),*];
//                     let b_input: [$ety; $ecount] = [$($in_b),*];
//                     let output: [$oty; $ecount] = [$($out),*];
//
//                     let a_vec_in: v128 = mem::transmute(a_input);
//                     let b_vec_in: v128 = mem::transmute(b_input);
//                     let vec_res: v128 = $id::$binary_op(a_vec_in, b_vec_in);
//
//                     let res: [$oty; $ecount] = mem::transmute(vec_res);
//
//                     if !floating_point!($ety) {
//                         assert_eq!(res, output);
//                     } else {
//                         for i in 0..$ecount {
//                             let r = res[i];
//                             let o = output[i];
//                             assert_eq!(r.is_nan(), o.is_nan());
//                             if !r.is_nan() {
//                                 assert_eq!(r, o);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//
//     macro_rules! test_bops {
//         ($id:ident[$ety:ident; $ecount:expr] |
//          $binary_op:ident [$op_test_id:ident]:
//          ([$($in_a:expr),*], $in_b:expr) => [$($out:expr),*]) => {
//             #[wasm_bindgen_test]
//             fn $op_test_id() {
//                 unsafe {
//                     let a_input: [$ety; $ecount] = [$($in_a),*];
//                     let output: [$ety; $ecount] = [$($out),*];
//
//                     let a_vec_in: v128 = mem::transmute(a_input);
//                     let vec_res: v128 = $id::$binary_op(a_vec_in, $in_b);
//
//                     let res: [$ety; $ecount] = mem::transmute(vec_res);
//                     assert_eq!(res, output);
//                 }
//             }
//         }
//     }
//
//     macro_rules! test_uop {
//         ($id:ident[$ety:ident; $ecount:expr] |
//          $unary_op:ident [$op_test_id:ident]: [$($in_a:expr),*] => [$($out:expr),*]) => {
//             #[wasm_bindgen_test]
//             fn $op_test_id() {
//                 unsafe {
//                     let a_input: [$ety; $ecount] = [$($in_a),*];
//                     let output: [$ety; $ecount] = [$($out),*];
//
//                     let a_vec_in: v128 = mem::transmute(a_input);
//                     let vec_res: v128 = $id::$unary_op(a_vec_in);
//
//                     let res: [$ety; $ecount] = mem::transmute(vec_res);
//                     assert_eq!(res, output);
//                 }
//             }
//         }
//     }
//
//     test_bop!(i8x16[i8; 16] | add[i8x16_add_test]:
//               ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1],
//                [8, i8::min_value(), 10, 11, 12, 13, 14, 1, 1, 1, 1, 1, 1, 1, 1, 1]) =>
//               [8, i8::max_value(), 12, 14, 16, 18, 20, i8::min_value(), 2, 2, 2, 2, 2, 2, 2, 2]);
//     test_bop!(i8x16[i8; 16] | sub[i8x16_sub_test]:
//               ([0, -1, 2, 3, 4, 5, 6, -1, 1, 1, 1, 1, 1, 1, 1, 1],
//                [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
//               [-8, i8::max_value(), -8, -8, -8, -8, -8, i8::min_value(), 0, 0, 0, 0, 0, 0, 0, 0]);
//     test_bop!(i8x16[i8; 16] | mul[i8x16_mul_test]:
//               ([0, -2, 2, 3, 4, 5, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1],
//                [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
//               [0, 0, 20, 33, 48, 65, 84, -2, 1, 1, 1, 1, 1, 1, 1, 1]);
//     test_uop!(i8x16[i8; 16] | neg[i8x16_neg_test]:
//               [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1] =>
//               [-8, i8::min_value(), -10, -11, -12, -13, -14, i8::min_value() + 1, -1, -1, -1, -1, -1, -1, -1, -1]);
//
//     test_bop!(i16x8[i16; 8] | add[i16x8_add_test]:
//               ([0, -1, 2, 3, 4, 5, 6, i16::max_value()],
//                [8, i16::min_value(), 10, 11, 12, 13, 14, 1]) =>
//               [8, i16::max_value(), 12, 14, 16, 18, 20, i16::min_value()]);
//     test_bop!(i16x8[i16; 8] | sub[i16x8_sub_test]:
//               ([0, -1, 2, 3, 4, 5, 6, -1],
//                [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
//               [-8, i16::max_value(), -8, -8, -8, -8, -8, i16::min_value()]);
//     test_bop!(i16x8[i16; 8] | mul[i16x8_mul_test]:
//               ([0, -2, 2, 3, 4, 5, 6, 2],
//                [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
//               [0, 0, 20, 33, 48, 65, 84, -2]);
//     test_uop!(i16x8[i16; 8] | neg[i16x8_neg_test]:
//               [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()] =>
//               [-8, i16::min_value(), -10, -11, -12, -13, -14, i16::min_value() + 1]);
//
//     test_bop!(i32x4[i32; 4] | add[i32x4_add_test]:
//               ([0, -1, 2, i32::max_value()],
//                [8, i32::min_value(), 10, 1]) =>
//               [8, i32::max_value(), 12, i32::min_value()]);
//     test_bop!(i32x4[i32; 4] | sub[i32x4_sub_test]:
//               ([0, -1, 2, -1],
//                [8, i32::min_value(), 10, i32::max_value()]) =>
//               [-8, i32::max_value(), -8, i32::min_value()]);
//     test_bop!(i32x4[i32; 4] | mul[i32x4_mul_test]:
//               ([0, -2, 2, 2],
//                [8, i32::min_value(), 10, i32::max_value()]) =>
//               [0, 0, 20, -2]);
//     test_uop!(i32x4[i32; 4] | neg[i32x4_neg_test]:
//               [8, i32::min_value(), 10, i32::max_value()] =>
//               [-8, i32::min_value(), -10, i32::min_value() + 1]);
//
//     test_bop!(i64x2[i64; 2] | add[i64x2_add_test]:
//               ([-1, i64::max_value()],
//                [i64::min_value(), 1]) =>
//               [i64::max_value(), i64::min_value()]);
//     test_bop!(i64x2[i64; 2] | sub[i64x2_sub_test]:
//               ([-1, -1],
//                [i64::min_value(), i64::max_value()]) =>
//               [ i64::max_value(), i64::min_value()]);
//     // note: mul for i64x2 is not part of the spec
//     test_uop!(i64x2[i64; 2] | neg[i64x2_neg_test]:
//               [i64::min_value(), i64::max_value()] =>
//               [i64::min_value(), i64::min_value() + 1]);
//
//     test_bops!(i8x16[i8; 16] | shl[i8x16_shl_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
//                [0, -2, 4, 6, 8, 10, 12, -2, 2, 2, 2, 2, 2, 2, 2, 2]);
//     test_bops!(i16x8[i16; 8] | shl[i16x8_shl_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
//                [0, -2, 4, 6, 8, 10, 12, -2]);
//     test_bops!(i32x4[i32; 4] | shl[i32x4_shl_test]:
//                ([0, -1, 2, 3], 1) => [0, -2, 4, 6]);
//     test_bops!(i64x2[i64; 2] | shl[i64x2_shl_test]:
//                ([0, -1], 1) => [0, -2]);
//
//     test_bops!(i8x16[i8; 16] | shr_s[i8x16_shr_s_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
//                [0, -1, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
//     test_bops!(i16x8[i16; 8] | shr_s[i16x8_shr_s_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
//                [0, -1, 1, 1, 2, 2, 3, i16::max_value() / 2]);
//     test_bops!(i32x4[i32; 4] | shr_s[i32x4_shr_s_test]:
//                ([0, -1, 2, 3], 1) => [0, -1, 1, 1]);
//     test_bops!(i64x2[i64; 2] | shr_s[i64x2_shr_s_test]:
//                ([0, -1], 1) => [0, -1]);
//
//     test_bops!(i8x16[i8; 16] | shr_u[i8x16_uhr_u_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
//                [0, i8::max_value(), 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
//     test_bops!(i16x8[i16; 8] | shr_u[i16x8_uhr_u_test]:
//                ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
//                [0, i16::max_value(), 1, 1, 2, 2, 3, i16::max_value() / 2]);
//     test_bops!(i32x4[i32; 4] | shr_u[i32x4_uhr_u_test]:
//                ([0, -1, 2, 3], 1) => [0, i32::max_value(), 1, 1]);
//     test_bops!(i64x2[i64; 2] | shr_u[i64x2_uhr_u_test]:
//                ([0, -1], 1) => [0, i64::max_value()]);
//
//     #[wasm_bindgen_test]
//     fn v128_bitwise_logical_ops() {
//         unsafe {
//             let a: [u32; 4] = [u32::max_value(), 0, u32::max_value(), 0];
//             let b: [u32; 4] = [u32::max_value(); 4];
//             let c: [u32; 4] = [0; 4];
//
//             let vec_a: v128 = mem::transmute(a);
//             let vec_b: v128 = mem::transmute(b);
//             let vec_c: v128 = mem::transmute(c);
//
//             let r: v128 = v128::and(vec_a, vec_a);
//             compare_bytes(r, vec_a);
//             let r: v128 = v128::and(vec_a, vec_b);
//             compare_bytes(r, vec_a);
//             let r: v128 = v128::or(vec_a, vec_b);
//             compare_bytes(r, vec_b);
//             let r: v128 = v128::not(vec_b);
//             compare_bytes(r, vec_c);
//             let r: v128 = v128::xor(vec_a, vec_c);
//             compare_bytes(r, vec_a);
//
//             let r: v128 = v128::bitselect(vec_b, vec_c, vec_b);
//             compare_bytes(r, vec_b);
//             let r: v128 = v128::bitselect(vec_b, vec_c, vec_c);
//             compare_bytes(r, vec_c);
//             let r: v128 = v128::bitselect(vec_b, vec_c, vec_a);
//             compare_bytes(r, vec_a);
//         }
//     }
//
//     macro_rules! test_bool_red {
//         ($id:ident[$test_id:ident] | [$($true:expr),*] | [$($false:expr),*] | [$($alt:expr),*]) => {
//             #[wasm_bindgen_test]
//             fn $test_id() {
//                 unsafe {
//                     let vec_a: v128 = mem::transmute([$($true),*]); // true
//                     let vec_b: v128 = mem::transmute([$($false),*]); // false
//                     let vec_c: v128 = mem::transmute([$($alt),*]); // alternating
//
//                     assert_eq!($id::any_true(vec_a), 1);
//                     assert_eq!($id::any_true(vec_b), 0);
//                     assert_eq!($id::any_true(vec_c), 1);
//
//                     assert_eq!($id::all_true(vec_a), 1);
//                     assert_eq!($id::all_true(vec_b), 0);
//                     assert_eq!($id::all_true(vec_c), 0);
//                 }
//             }
//         }
//     }
//
//     test_bool_red!(
//         i8x16[i8x16_boolean_reductions]
//             | [1_i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
//             | [0_i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//             | [1_i8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
//     );
//     test_bool_red!(
//         i16x8[i16x8_boolean_reductions]
//             | [1_i16, 1, 1, 1, 1, 1, 1, 1]
//             | [0_i16, 0, 0, 0, 0, 0, 0, 0]
//             | [1_i16, 0, 1, 0, 1, 0, 1, 0]
//     );
//     test_bool_red!(
//         i32x4[i32x4_boolean_reductions]
//             | [1_i32, 1, 1, 1]
//             | [0_i32, 0, 0, 0]
//             | [1_i32, 0, 1, 0]
//     );
//     test_bool_red!(
//         i64x2[i64x2_boolean_reductions] | [1_i64, 1] | [0_i64, 0] | [1_i64, 0]
//     );
//
//     test_bop!(i8x16[i8; 16] | eq[i8x16_eq_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i16x8[i16; 8] | eq[i16x8_eq_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i32x4[i32; 4] | eq[i32x4_eq_test]:
//               ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
//     test_bop!(i64x2[i64; 2] | eq[i64x2_eq_test]: ([0, 1], [0, 2]) => [-1, 0]);
//     test_bop!(f32x4[f32; 4] => i32 | eq[f32x4_eq_test]:
//               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
//     test_bop!(f64x2[f64; 2] => i64 | eq[f64x2_eq_test]: ([0., 1.], [0., 2.]) => [-1, 0]);
//
//     test_bop!(i8x16[i8; 16] | ne[i8x16_ne_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i16x8[i16; 8] | ne[i16x8_ne_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i32x4[i32; 4] | ne[i32x4_ne_test]:
//               ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
//     test_bop!(i64x2[i64; 2] | ne[i64x2_ne_test]: ([0, 1], [0, 2]) => [0, -1]);
//     test_bop!(f32x4[f32; 4] => i32 | ne[f32x4_ne_test]:
//               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
//     test_bop!(f64x2[f64; 2] => i64 | ne[f64x2_ne_test]: ([0., 1.], [0., 2.]) => [0, -1]);
//
//     test_bop!(i8x16[i8; 16] | lt[i8x16_lt_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i16x8[i16; 8] | lt[i16x8_lt_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i32x4[i32; 4] | lt[i32x4_lt_test]:
//               ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
//     test_bop!(i64x2[i64; 2] | lt[i64x2_lt_test]: ([0, 1], [0, 2]) => [0, -1]);
//     test_bop!(f32x4[f32; 4] => i32 | lt[f32x4_lt_test]:
//               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
//     test_bop!(f64x2[f64; 2] => i64 | lt[f64x2_lt_test]: ([0., 1.], [0., 2.]) => [0, -1]);
//
//     test_bop!(i8x16[i8; 16] | gt[i8x16_gt_test]:
//           ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
//            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i16x8[i16; 8] | gt[i16x8_gt_test]:
//               ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
//               [0, -1, 0, -1 ,0, -1, 0, 0]);
//     test_bop!(i32x4[i32; 4] | gt[i32x4_gt_test]:
//               ([0, 2, 2, 4], [0, 1, 2, 3]) => [0, -1, 0, -1]);
//     test_bop!(i64x2[i64; 2] | gt[i64x2_gt_test]: ([0, 2], [0, 1]) => [0, -1]);
//     test_bop!(f32x4[f32; 4] => i32 | gt[f32x4_gt_test]:
//               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [0, -1, 0, -1]);
//     test_bop!(f64x2[f64; 2] => i64 | gt[f64x2_gt_test]: ([0., 2.], [0., 1.]) => [0, -1]);
//
//     test_bop!(i8x16[i8; 16] | ge[i8x16_ge_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i16x8[i16; 8] | ge[i16x8_ge_test]:
//               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i32x4[i32; 4] | ge[i32x4_ge_test]:
//               ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
//     test_bop!(i64x2[i64; 2] | ge[i64x2_ge_test]: ([0, 1], [0, 2]) => [-1, 0]);
//     test_bop!(f32x4[f32; 4] => i32 | ge[f32x4_ge_test]:
//               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
//     test_bop!(f64x2[f64; 2] => i64 | ge[f64x2_ge_test]: ([0., 1.], [0., 2.]) => [-1, 0]);
//
//     test_bop!(i8x16[i8; 16] | le[i8x16_le_test]:
//               ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
//                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//               ) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i16x8[i16; 8] | le[i16x8_le_test]:
//               ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
//               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
//     test_bop!(i32x4[i32; 4] | le[i32x4_le_test]:
//               ([0, 2, 2, 4], [0, 1, 2, 3]) => [-1, 0, -1, 0]);
//     test_bop!(i64x2[i64; 2] | le[i64x2_le_test]: ([0, 2], [0, 1]) => [-1, 0]);
//     test_bop!(f32x4[f32; 4] => i32 | le[f32x4_le_test]:
//               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [-1, 0, -1, -0]);
//     test_bop!(f64x2[f64; 2] => i64 | le[f64x2_le_test]: ([0., 2.], [0., 1.]) => [-1, 0]);
//
//     #[wasm_bindgen_test]
//     fn v128_bitwise_load_store() {
//         unsafe {
//             let mut arr: [i32; 4] = [0, 1, 2, 3];
//
//             let vec = v128::load(arr.as_ptr() as *const v128);
//             let vec = i32x4::add(vec, vec);
//             v128::store(arr.as_mut_ptr() as *mut v128, vec);
//
//             assert_eq!(arr, [0, 2, 4, 6]);
//         }
//     }
//
//     test_uop!(f32x4[f32; 4] | neg[f32x4_neg_test]: [0., 1., 2., 3.] => [ 0., -1., -2., -3.]);
//     test_uop!(f32x4[f32; 4] | abs[f32x4_abs_test]: [0., -1., 2., -3.] => [ 0., 1., 2., 3.]);
//     test_bop!(f32x4[f32; 4] | min[f32x4_min_test]:
//               ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., -3., -4., 8.]);
//     test_bop!(f32x4[f32; 4] | min[f32x4_min_test_nan]:
//               ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
//               => [0., -3., -4., std::f32::NAN]);
//     test_bop!(f32x4[f32; 4] | max[f32x4_max_test]:
//               ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -1., 7., 10.]);
//     test_bop!(f32x4[f32; 4] | max[f32x4_max_test_nan]:
//               ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
//               => [1., -1., 7., std::f32::NAN]);
//     test_bop!(f32x4[f32; 4] | add[f32x4_add_test]:
//               ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -4., 3., 18.]);
//     test_bop!(f32x4[f32; 4] | sub[f32x4_sub_test]:
//               ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [-1., 2., 11., -2.]);
//     test_bop!(f32x4[f32; 4] | mul[f32x4_mul_test]:
//               ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., 3., -28., 80.]);
//     test_bop!(f32x4[f32; 4] | div[f32x4_div_test]:
//               ([0., -8., 70., 8.], [1., 4., 10., 2.]) => [0., -2., 7., 4.]);
//
//     test_uop!(f64x2[f64; 2] | neg[f64x2_neg_test]: [0., 1.] => [ 0., -1.]);
//     test_uop!(f64x2[f64; 2] | abs[f64x2_abs_test]: [0., -1.] => [ 0., 1.]);
//     test_bop!(f64x2[f64; 2] | min[f64x2_min_test]:
//               ([0., -1.], [1., -3.]) => [0., -3.]);
//     test_bop!(f64x2[f64; 2] | min[f64x2_min_test_nan]:
//               ([7., 8.], [-4., std::f64::NAN])
//               => [ -4., std::f64::NAN]);
//     test_bop!(f64x2[f64; 2] | max[f64x2_max_test]:
//               ([0., -1.], [1., -3.]) => [1., -1.]);
//     test_bop!(f64x2[f64; 2] | max[f64x2_max_test_nan]:
//               ([7., 8.], [ -4., std::f64::NAN])
//               => [7., std::f64::NAN]);
//     test_bop!(f64x2[f64; 2] | add[f64x2_add_test]:
//               ([0., -1.], [1., -3.]) => [1., -4.]);
//     test_bop!(f64x2[f64; 2] | sub[f64x2_sub_test]:
//               ([0., -1.], [1., -3.]) => [-1., 2.]);
//     test_bop!(f64x2[f64; 2] | mul[f64x2_mul_test]:
//               ([0., -1.], [1., -3.]) => [0., 3.]);
//     test_bop!(f64x2[f64; 2] | div[f64x2_div_test]:
//               ([0., -8.], [1., 4.]) => [0., -2.]);
//
//     macro_rules! test_conv {
//         ($test_id:ident | $conv_id:ident | $to_ty:ident | $from:expr,  $to:expr) => {
//             #[wasm_bindgen_test]
//             fn $test_id() {
//                 unsafe {
//                     let from: v128 = mem::transmute($from);
//                     let to: v128 = mem::transmute($to);
//
//                     let r: v128 = $to_ty::$conv_id(from);
//
//                     compare_bytes(r, to);
//                 }
//             }
//         };
//     }
//
//     test_conv!(
//         f32x4_convert_s_i32x4 | convert_s_i32x4 | f32x4 | [1_i32, 2, 3, 4],
//         [1_f32, 2., 3., 4.]
//     );
//     test_conv!(
//         f32x4_convert_u_i32x4
//             | convert_u_i32x4
//             | f32x4
//             | [u32::max_value(), 2, 3, 4],
//         [u32::max_value() as f32, 2., 3., 4.]
//     );
//     test_conv!(
//         f64x2_convert_s_i64x2 | convert_s_i64x2 | f64x2 | [1_i64, 2],
//         [1_f64, 2.]
//     );
//     test_conv!(
//         f64x2_convert_u_i64x2
//             | convert_u_i64x2
//             | f64x2
//             | [u64::max_value(), 2],
//         [18446744073709552000.0, 2.]
//     );
//
//     // FIXME: this fails, and produces -2147483648 instead of saturating at
//     // i32::max_value() test_conv!(i32x4_trunc_s_f32x4_sat | trunc_s_f32x4_sat
//     // | i32x4 | [1_f32, 2., (i32::max_value() as f32 + 1.), 4.],
//     // [1_i32, 2, i32::max_value(), 4]); FIXME: add other saturating tests
// }
