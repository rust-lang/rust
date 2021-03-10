// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen/neon.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
// ```
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_and(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_or(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_xor(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v8i8")]
        fn vabd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vabd_s8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v16i8")]
        fn vabdq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vabdq_s8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v4i16")]
        fn vabd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vabd_s16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v8i16")]
        fn vabdq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vabdq_s16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v2i32")]
        fn vabd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vabd_s32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v4i32")]
        fn vabdq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vabdq_s32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v8i8")]
        fn vabd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vabd_u8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v16i8")]
        fn vabdq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vabdq_u8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v4i16")]
        fn vabd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vabd_u16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v8i16")]
        fn vabdq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vabdq_u16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v2i32")]
        fn vabd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vabd_u32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v4i32")]
        fn vabdq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vabdq_u32_(a, b)
}

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabd))]
pub unsafe fn vabd_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v2f32")]
        fn vabd_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vabd_f32_(a, b)
}

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabd))]
pub unsafe fn vabdq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v4f32")]
        fn vabdq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vabdq_f32_(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_p8(a: poly8x8_t, b: poly8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_p8(a: poly8x16_t, b: poly8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceq_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceqq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    let c: int8x8_t = simd_and(a, b);
    let d: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    let c: int8x16_t = simd_and(a, b);
    let d: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    let c: int16x4_t = simd_and(a, b);
    let d: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    let c: int16x8_t = simd_and(a, b);
    let d: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    let c: int32x2_t = simd_and(a, b);
    let d: i32x2 = i32x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    let c: int32x4_t = simd_and(a, b);
    let d: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_p8(a: poly8x8_t, b: poly8x8_t) -> uint8x8_t {
    let c: poly8x8_t = simd_and(a, b);
    let d: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_p8(a: poly8x16_t, b: poly8x16_t) -> uint8x16_t {
    let c: poly8x16_t = simd_and(a, b);
    let d: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let c: uint8x8_t = simd_and(a, b);
    let d: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let c: uint8x16_t = simd_and(a, b);
    let d: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    let c: uint16x4_t = simd_and(a, b);
    let d: u16x4 = u16x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let c: uint16x8_t = simd_and(a, b);
    let d: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    let c: uint32x2_t = simd_and(a, b);
    let d: u32x2 = u32x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    let c: uint32x4_t = simd_and(a, b);
    let d: u32x4 = u32x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vabs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabs))]
pub unsafe fn vabs_f32(a: float32x2_t) -> float32x2_t {
    simd_fabs(a)
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vabs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabs))]
pub unsafe fn vabsq_f32(a: float32x4_t) -> float32x4_t {
    simd_fabs(a)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vclt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcltq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcle_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcleq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcge_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcgeq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i8")]
        fn vqsub_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vqsub_u8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v16i8")]
        fn vqsubq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vqsubq_u8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i16")]
        fn vqsub_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vqsub_u16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i16")]
        fn vqsubq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vqsubq_u16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v2i32")]
        fn vqsub_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vqsub_u32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i32")]
        fn vqsubq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vqsubq_u32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i8")]
        fn vqsub_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqsub_s8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v16i8")]
        fn vqsubq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqsubq_s8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i16")]
        fn vqsub_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqsub_s16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i16")]
        fn vqsubq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqsubq_s16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v2i32")]
        fn vqsub_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqsub_s32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqsubs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i32")]
        fn vqsubq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqsubq_s32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i8")]
        fn vhadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vhadd_u8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v16i8")]
        fn vhaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vhaddq_u8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i16")]
        fn vhadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vhadd_u16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i16")]
        fn vhaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vhaddq_u16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v2i32")]
        fn vhadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vhadd_u32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i32")]
        fn vhaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vhaddq_u32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i8")]
        fn vhadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vhadd_s8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v16i8")]
        fn vhaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vhaddq_s8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i16")]
        fn vhadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vhadd_s16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i16")]
        fn vhaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vhaddq_s16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v2i32")]
        fn vhadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vhadd_s32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i32")]
        fn vhaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vhaddq_s32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v8i8")]
        fn vrhadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vrhadd_u8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v16i8")]
        fn vrhaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vrhaddq_u8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v4i16")]
        fn vrhadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vrhadd_u16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v8i16")]
        fn vrhaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vrhaddq_u16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v2i32")]
        fn vrhadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vrhadd_u32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v4i32")]
        fn vrhaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vrhaddq_u32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v8i8")]
        fn vrhadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vrhadd_s8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v16i8")]
        fn vrhaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vrhaddq_s8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v4i16")]
        fn vrhadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vrhadd_s16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v8i16")]
        fn vrhaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vrhaddq_s16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v2i32")]
        fn vrhadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vrhadd_s32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v4i32")]
        fn vrhaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vrhaddq_s32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i8")]
        fn vqadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vqadd_u8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v16i8")]
        fn vqaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vqaddq_u8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i16")]
        fn vqadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vqadd_u16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i16")]
        fn vqaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vqaddq_u16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v2i32")]
        fn vqadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vqadd_u32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqaddu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i32")]
        fn vqaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vqaddq_u32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i8")]
        fn vqadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqadd_s8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v16i8")]
        fn vqaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqaddq_s8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i16")]
        fn vqadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqadd_s16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i16")]
        fn vqaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqaddq_s16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v2i32")]
        fn vqadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqadd_s32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqadds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i32")]
        fn vqaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqaddq_s32_(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmul_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmulq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_mul(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fsub))]
pub unsafe fn vsub_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fsub))]
pub unsafe fn vsubq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_sub(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v8i8")]
        fn vhsub_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vhsub_u8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v16i8")]
        fn vhsubq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vhsubq_u8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v4i16")]
        fn vhsub_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vhsub_u16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v8i16")]
        fn vhsubq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vhsubq_u16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v2i32")]
        fn vhsub_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vhsub_u32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v4i32")]
        fn vhsubq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vhsubq_u32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v8i8")]
        fn vhsub_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vhsub_s8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v16i8")]
        fn vhsubq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vhsubq_s8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v4i16")]
        fn vhsub_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vhsub_s16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v8i16")]
        fn vhsubq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vhsubq_s16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v2i32")]
        fn vhsub_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vhsub_s32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v4i32")]
        fn vhsubq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vhsubq_s32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v8i8")]
        fn vmax_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vmax_s8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v16i8")]
        fn vmaxq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vmaxq_s8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v4i16")]
        fn vmax_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vmax_s16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v8i16")]
        fn vmaxq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vmaxq_s16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v2i32")]
        fn vmax_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vmax_s32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v4i32")]
        fn vmaxq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vmaxq_s32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v8i8")]
        fn vmax_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vmax_u8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v16i8")]
        fn vmaxq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vmaxq_u8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v4i16")]
        fn vmax_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vmax_u16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v8i16")]
        fn vmaxq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vmaxq_u16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v2i32")]
        fn vmax_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vmax_u32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v4i32")]
        fn vmaxq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vmaxq_u32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmax))]
pub unsafe fn vmax_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v2f32")]
        fn vmax_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vmax_f32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmax))]
pub unsafe fn vmaxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v4f32")]
        fn vmaxq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vmaxq_f32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v8i8")]
        fn vmin_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vmin_s8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v16i8")]
        fn vminq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vminq_s8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v4i16")]
        fn vmin_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vmin_s16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v8i16")]
        fn vminq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vminq_s16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v2i32")]
        fn vmin_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vmin_s32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v4i32")]
        fn vminq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vminq_s32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v8i8")]
        fn vmin_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vmin_u8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v16i8")]
        fn vminq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vminq_u8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v4i16")]
        fn vmin_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vmin_u16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v8i16")]
        fn vminq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vminq_u16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v2i32")]
        fn vmin_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vmin_u32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v4i32")]
        fn vminq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vminq_u32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmin))]
pub unsafe fn vmin_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v2f32")]
        fn vmin_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vmin_f32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmin))]
pub unsafe fn vminq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v4f32")]
        fn vminq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vminq_f32_(a, b)
}

#[cfg(test)]
#[allow(overflowing_literals)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: i8x16 = i8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x0F, 0x0F);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x00);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: u8x16 = u8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x0F, 0x0F);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x00);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x0F);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x0F, 0x0F);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x00);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x0F);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x0F, 0x0F);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x00);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vorr_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(vorrq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vorr_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vorrq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vorr_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vorrq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vorr_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(vorrq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vorr_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vorrq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vorr_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vorrq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vorr_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vorrq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vorr_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vorrq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(veor_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(veorq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(veor_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(veorq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(veor_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(veorq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(veor_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(veorq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(veor_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(veorq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(veor_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(veorq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(veor_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(veorq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(veor_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(veorq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: i8x8 = transmute(vabd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(15, 13, 11, 9, 7, 5, 3, 1, 1, 3, 5, 7, 9, 11, 13, 15);
        let r: i8x16 = transmute(vabdq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(15, 13, 11, 9);
        let r: i16x4 = transmute(vabd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: i16x8 = transmute(vabdq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(15, 13);
        let r: i32x2 = transmute(vabd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(15, 13, 11, 9);
        let r: i32x4 = transmute(vabdq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: u8x8 = transmute(vabd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(15, 13, 11, 9, 7, 5, 3, 1, 1, 3, 5, 7, 9, 11, 13, 15);
        let r: u8x16 = transmute(vabdq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(15, 13, 11, 9);
        let r: u16x4 = transmute(vabd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: u16x8 = transmute(vabdq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(15, 13);
        let r: u32x2 = transmute(vabd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(15, 13, 11, 9);
        let r: u32x4 = transmute(vabdq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(9.0, 3.0);
        let e: f32x2 = f32x2::new(8.0, 1.0);
        let r: f32x2 = transmute(vabd_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 5.0, -4.0);
        let b: f32x4 = f32x4::new(9.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(8.0, 1.0, 3.0, 12.0);
        let r: f32x4 = transmute(vabdq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        let a: u8x8 = u8x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x8 = u8x8::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0, 0xFF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        let a: u8x16 = u8x16::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0xFF);
        let b: u8x16 = u8x16::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0xFF);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x16 = u8x16::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0xFF);
        let b: u8x16 = u8x16::new(0, 0xFF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, 0);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        let a: u16x4 = u16x4::new(0, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0, 0x01, 0x02, 0x03);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x4 = u16x4::new(0, 0, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0, 0xFF_FF, 0x02, 0x04);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        let a: u16x8 = u16x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x8 = u16x8::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0, 0xFF_FF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        let a: u32x2 = u32x2::new(0, 0x01);
        let b: u32x2 = u32x2::new(0, 0x01);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x2 = u32x2::new(0, 0);
        let b: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        let a: u32x4 = u32x4::new(0, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0, 0x01, 0x02, 0x03);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x4 = u32x4::new(0, 0, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0x02, 0x04);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        let a: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, -128);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(-32768, 0x01, 0x02, 0x03);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x4 = i16x4::new(-32768, -32768, 0x02, 0x03);
        let b: i16x4 = i16x4::new(-32768, 0x7F_FF, 0x02, 0x04);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x4 = transmute(vceq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(-32768, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x8 = i16x8::new(-32768, -32768, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(-32768, 0x7F_FF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x8 = transmute(vceqq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x01);
        let b: i32x2 = i32x2::new(-2147483648, 0x01);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x2 = i32x2::new(-2147483648, -2147483648);
        let b: i32x2 = i32x2::new(-2147483648, 0x7F_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vceq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(-2147483648, 0x01, 0x02, 0x03);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x4 = i32x4::new(-2147483648, -2147483648, 0x02, 0x03);
        let b: i32x4 = i32x4::new(-2147483648, 0x7F_FF_FF_FF, 0x02, 0x04);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vceqq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_p8() {
        let a: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_p8() {
        let a: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, -128);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        let a: f32x2 = f32x2::new(1.2, 3.4);
        let b: f32x2 = f32x2::new(1.2, 3.4);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        let a: f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b: f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x00, 0x01, 0x02);
        let b: i16x4 = i16x4::new(-32768, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vtst_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i16x8 = i16x8::new(-32768, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vtstq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x00);
        let b: i32x2 = i32x2::new(-2147483648, 0x00);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vtst_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x00, 0x01, 0x02);
        let b: i32x4 = i32x4::new(-2147483648, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vtstq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_p8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_p8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u8() {
        let a: u8x8 = u8x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: u8x8 = u8x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u8() {
        let a: u8x16 = u8x16::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFF);
        let b: u8x16 = u8x16::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFF);
        let e: u8x16 = u8x16::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u16() {
        let a: u16x4 = u16x4::new(0, 0x00, 0x01, 0x02);
        let b: u16x4 = u16x4::new(0, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0, 0, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vtst_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u16() {
        let a: u16x8 = u16x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: u16x8 = u16x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vtstq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u32() {
        let a: u32x2 = u32x2::new(0, 0x00);
        let b: u32x2 = u32x2::new(0, 0x00);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vtst_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u32() {
        let a: u32x4 = u32x4::new(0, 0x00, 0x01, 0x02);
        let b: u32x4 = u32x4::new(0, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vtstq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabs_f32() {
        let a: f32x2 = f32x2::new(-0.1, -2.2);
        let e: f32x2 = f32x2::new(0.1, 2.2);
        let r: f32x2 = transmute(vabs_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabsq_f32() {
        let a: f32x4 = f32x4::new(-0.1, -2.2, -3.3, -6.6);
        let e: f32x4 = f32x4::new(0.1, 2.2, 3.3, 6.6);
        let r: f32x4 = transmute(vabsq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        let a: f32x2 = f32x2::new(1.2, 2.3);
        let b: f32x2 = f32x2::new(0.1, 1.2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        let a: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        let a: f32x2 = f32x2::new(0.1, 1.2);
        let b: f32x2 = f32x2::new(1.2, 2.3);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        let a: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        let a: f32x2 = f32x2::new(0.1, 1.2);
        let b: f32x2 = f32x2::new(1.2, 2.3);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        let a: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        let a: f32x2 = f32x2::new(1.2, 2.3);
        let b: f32x2 = f32x2::new(0.1, 1.2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        let a: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u8x8 = transmute(vqsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: u8x16 = transmute(vqsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(41, 40, 39, 38);
        let r: u16x4 = transmute(vqsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u16x8 = transmute(vqsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(41, 40);
        let r: u32x2 = transmute(vqsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(41, 40, 39, 38);
        let r: u32x4 = transmute(vqsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i8x8 = transmute(vqsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: i8x16 = transmute(vqsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(41, 40, 39, 38);
        let r: i16x4 = transmute(vqsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i16x8 = transmute(vqsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(41, 40);
        let r: i32x2 = transmute(vqsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(41, 40, 39, 38);
        let r: i32x4 = transmute(vqsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u8x8 = transmute(vhadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: u8x16 = transmute(vhaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(21, 22, 22, 23);
        let r: u16x4 = transmute(vhadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u16x8 = transmute(vhaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(21, 22);
        let r: u32x2 = transmute(vhadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(21, 22, 22, 23);
        let r: u32x4 = transmute(vhaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i8x8 = transmute(vhadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: i8x16 = transmute(vhaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(21, 22, 22, 23);
        let r: i16x4 = transmute(vhadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i16x8 = transmute(vhaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(21, 22);
        let r: i32x2 = transmute(vhadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(21, 22, 22, 23);
        let r: i32x4 = transmute(vhaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: u8x8 = transmute(vrhadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29);
        let r: u8x16 = transmute(vrhaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(22, 22, 23, 23);
        let r: u16x4 = transmute(vrhadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: u16x8 = transmute(vrhaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(22, 22);
        let r: u32x2 = transmute(vrhadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(22, 22, 23, 23);
        let r: u32x4 = transmute(vrhaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: i8x8 = transmute(vrhadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29);
        let r: i8x16 = transmute(vrhaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(22, 22, 23, 23);
        let r: i16x4 = transmute(vrhadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: i16x8 = transmute(vrhaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(22, 22);
        let r: i32x2 = transmute(vrhadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(22, 22, 23, 23);
        let r: i32x4 = transmute(vrhaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u8x8 = transmute(vqadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: u8x16 = transmute(vqaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(43, 44, 45, 46);
        let r: u16x4 = transmute(vqadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u16x8 = transmute(vqaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(43, 44);
        let r: u32x2 = transmute(vqadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(43, 44, 45, 46);
        let r: u32x4 = transmute(vqaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i8x8 = transmute(vqadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: i8x16 = transmute(vqaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(43, 44, 45, 46);
        let r: i16x4 = transmute(vqadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i16x8 = transmute(vqaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(43, 44);
        let r: i32x2 = transmute(vqadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(43, 44, 45, 46);
        let r: i32x4 = transmute(vqaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s8() {
        let a: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: i8x8 = transmute(vmul_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24, 13, 28, 15, 32);
        let r: i8x16 = transmute(vmulq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s16() {
        let a: i16x4 = i16x4::new(1, 2, 1, 2);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(1, 4, 3, 8);
        let r: i16x4 = transmute(vmul_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: i16x8 = transmute(vmulq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(1, 4);
        let r: i32x2 = transmute(vmul_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 1, 2);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(1, 4, 3, 8);
        let r: i32x4 = transmute(vmulq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u8() {
        let a: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: u8x8 = transmute(vmul_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24, 13, 28, 15, 32);
        let r: u8x16 = transmute(vmulq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u16() {
        let a: u16x4 = u16x4::new(1, 2, 1, 2);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(1, 4, 3, 8);
        let r: u16x4 = transmute(vmul_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: u16x8 = transmute(vmulq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(1, 4);
        let r: u32x2 = transmute(vmul_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 1, 2);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(1, 4, 3, 8);
        let r: u32x4 = transmute(vmulq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(2.0, 3.0);
        let e: f32x2 = f32x2::new(2.0, 6.0);
        let r: f32x2 = transmute(vmul_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 1.0, 2.0);
        let b: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let e: f32x4 = f32x4::new(2.0, 6.0, 4.0, 10.0);
        let r: f32x4 = transmute(vmulq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x8 = i8x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: i8x8 = transmute(vsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x16 = i8x16::new(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
        let r: i8x16 = transmute(vsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 1, 2);
        let e: i16x4 = i16x4::new(0, 0, 2, 2);
        let r: i16x4 = transmute(vsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: i16x8 = transmute(vsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(1, 2, 1, 2);
        let e: i32x4 = i32x4::new(0, 0, 2, 2);
        let r: i32x4 = transmute(vsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x8 = u8x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u8x8 = transmute(vsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x16 = u8x16::new(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
        let r: u8x16 = transmute(vsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 1, 2);
        let e: u16x4 = u16x4::new(0, 0, 2, 2);
        let r: u16x4 = transmute(vsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u16x8 = transmute(vsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(1, 2, 1, 2);
        let e: u32x4 = u32x4::new(0, 0, 2, 2);
        let r: u32x4 = transmute(vsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vsub_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(1, 2);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vsubq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vsub_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vsubq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f32() {
        let a: f32x2 = f32x2::new(1.0, 4.0);
        let b: f32x2 = f32x2::new(1.0, 2.0);
        let e: f32x2 = f32x2::new(0.0, 2.0);
        let r: f32x2 = transmute(vsub_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f32() {
        let a: f32x4 = f32x4::new(1.0, 4.0, 3.0, 8.0);
        let b: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let e: f32x4 = f32x4::new(0.0, 2.0, 0.0, 4.0);
        let r: f32x4 = transmute(vsubq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x8 = u8x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: u8x8 = transmute(vhsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x16 = u8x16::new(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let r: u8x16 = transmute(vhsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 1, 2);
        let e: u16x4 = u16x4::new(0, 0, 1, 1);
        let r: u16x4 = transmute(vhsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: u16x8 = transmute(vhsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vhsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(1, 2, 1, 2);
        let e: u32x4 = u32x4::new(0, 0, 1, 1);
        let r: u32x4 = transmute(vhsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x8 = i8x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: i8x8 = transmute(vhsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x16 = i8x16::new(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let r: i8x16 = transmute(vhsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 1, 2);
        let e: i16x4 = i16x4::new(0, 0, 1, 1);
        let r: i16x4 = transmute(vhsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: i16x8 = transmute(vhsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vhsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(1, 2, 1, 2);
        let e: i32x4 = i32x4::new(0, 0, 1, 1);
        let r: i32x4 = transmute(vhsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: i8x8 = transmute(vmax_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vmaxq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(16, 15, 14, 13);
        let r: i16x4 = transmute(vmax_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: i16x8 = transmute(vmaxq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(16, 15);
        let r: i32x2 = transmute(vmax_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(16, 15, 14, 13);
        let r: i32x4 = transmute(vmaxq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: u8x8 = transmute(vmax_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vmaxq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(16, 15, 14, 13);
        let r: u16x4 = transmute(vmax_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: u16x8 = transmute(vmaxq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(16, 15);
        let r: u32x2 = transmute(vmax_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(16, 15, 14, 13);
        let r: u32x4 = transmute(vmaxq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_f32() {
        let a: f32x2 = f32x2::new(1.0, -2.0);
        let b: f32x2 = f32x2::new(0.0, 3.0);
        let e: f32x2 = f32x2::new(1.0, 3.0);
        let r: f32x2 = transmute(vmax_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_f32() {
        let a: f32x4 = f32x4::new(1.0, -2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(0.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(1.0, 3.0, 3.0, 8.0);
        let r: f32x4 = transmute(vmaxq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vmin_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1);
        let r: i8x16 = transmute(vminq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vmin_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vminq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vmin_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vminq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vmin_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1);
        let r: u8x16 = transmute(vminq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vmin_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vminq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vmin_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vminq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_f32() {
        let a: f32x2 = f32x2::new(1.0, -2.0);
        let b: f32x2 = f32x2::new(0.0, 3.0);
        let e: f32x2 = f32x2::new(0.0, -2.0);
        let r: f32x2 = transmute(vmin_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_f32() {
        let a: f32x4 = f32x4::new(1.0, -2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(0.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(0.0, -2.0, 2.0, -4.0);
        let r: f32x4 = transmute(vminq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
}
