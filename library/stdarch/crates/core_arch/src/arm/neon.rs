use crate::core_arch::arm_shared::neon::*;
use crate::core_arch::simd::{f32x4, i32x4, u32x4};
use crate::core_arch::simd_llvm::*;
use crate::mem::{align_of, transmute};

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(non_camel_case_types)]
pub(crate) type p8 = u8;
#[allow(non_camel_case_types)]
pub(crate) type p16 = u16;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.arm.neon.vbsl.v8i8"]
    fn vbsl_s8_(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vbsl.v16i8"]
    fn vbslq_s8_(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
    #[link_name = "llvm.arm.neon.vpadals.v4i16.v8i8"]
    pub(crate) fn vpadal_s8_(a: int16x4_t, b: int8x8_t) -> int16x4_t;
    #[link_name = "llvm.arm.neon.vpadals.v2i32.v4i16"]
    pub(crate) fn vpadal_s16_(a: int32x2_t, b: int16x4_t) -> int32x2_t;
    #[link_name = "llvm.arm.neon.vpadals.v1i64.v2i32"]
    pub(crate) fn vpadal_s32_(a: int64x1_t, b: int32x2_t) -> int64x1_t;
    #[link_name = "llvm.arm.neon.vpadals.v8i16.v16i8"]
    pub(crate) fn vpadalq_s8_(a: int16x8_t, b: int8x16_t) -> int16x8_t;
    #[link_name = "llvm.arm.neon.vpadals.v4i32.v8i16"]
    pub(crate) fn vpadalq_s16_(a: int32x4_t, b: int16x8_t) -> int32x4_t;
    #[link_name = "llvm.arm.neon.vpadals.v2i64.v4i32"]
    pub(crate) fn vpadalq_s32_(a: int64x2_t, b: int32x4_t) -> int64x2_t;

    #[link_name = "llvm.arm.neon.vpadalu.v4i16.v8i8"]
    pub(crate) fn vpadal_u8_(a: uint16x4_t, b: uint8x8_t) -> uint16x4_t;
    #[link_name = "llvm.arm.neon.vpadalu.v2i32.v4i16"]
    pub(crate) fn vpadal_u16_(a: uint32x2_t, b: uint16x4_t) -> uint32x2_t;
    #[link_name = "llvm.arm.neon.vpadalu.v1i64.v2i32"]
    pub(crate) fn vpadal_u32_(a: uint64x1_t, b: uint32x2_t) -> uint64x1_t;
    #[link_name = "llvm.arm.neon.vpadalu.v8i16.v16i8"]
    pub(crate) fn vpadalq_u8_(a: uint16x8_t, b: uint8x16_t) -> uint16x8_t;
    #[link_name = "llvm.arm.neon.vpadalu.v4i32.v8i16"]
    pub(crate) fn vpadalq_u16_(a: uint32x4_t, b: uint16x8_t) -> uint32x4_t;
    #[link_name = "llvm.arm.neon.vpadalu.v2i64.v4i32"]
    pub(crate) fn vpadalq_u32_(a: uint64x2_t, b: uint32x4_t) -> uint64x2_t;

    #[link_name = "llvm.arm.neon.vtbl1"]
    fn vtbl1(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl2"]
    fn vtbl2(a: int8x8_t, b: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl3"]
    fn vtbl3(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbl4"]
    fn vtbl4(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t, d: int8x8_t) -> int8x8_t;

    #[link_name = "llvm.arm.neon.vtbx1"]
    fn vtbx1(a: int8x8_t, b: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx2"]
    fn vtbx2(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx3"]
    fn vtbx3(a: int8x8_t, b: int8x8_t, b: int8x8_t, c: int8x8_t, d: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vtbx4"]
    fn vtbx4(
        a: int8x8_t,
        b: int8x8_t,
        b: int8x8_t,
        c: int8x8_t,
        d: int8x8_t,
        e: int8x8_t,
    ) -> int8x8_t;

    #[link_name = "llvm.arm.neon.vshiftins.v8i8"]
    fn vshiftins_v8i8(a: int8x8_t, b: int8x8_t, shift: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vshiftins.v16i8"]
    fn vshiftins_v16i8(a: int8x16_t, b: int8x16_t, shift: int8x16_t) -> int8x16_t;
    #[link_name = "llvm.arm.neon.vshiftins.v4i16"]
    fn vshiftins_v4i16(a: int16x4_t, b: int16x4_t, shift: int16x4_t) -> int16x4_t;
    #[link_name = "llvm.arm.neon.vshiftins.v8i16"]
    fn vshiftins_v8i16(a: int16x8_t, b: int16x8_t, shift: int16x8_t) -> int16x8_t;
    #[link_name = "llvm.arm.neon.vshiftins.v2i32"]
    fn vshiftins_v2i32(a: int32x2_t, b: int32x2_t, shift: int32x2_t) -> int32x2_t;
    #[link_name = "llvm.arm.neon.vshiftins.v4i32"]
    fn vshiftins_v4i32(a: int32x4_t, b: int32x4_t, shift: int32x4_t) -> int32x4_t;
    #[link_name = "llvm.arm.neon.vshiftins.v1i64"]
    fn vshiftins_v1i64(a: int64x1_t, b: int64x1_t, shift: int64x1_t) -> int64x1_t;
    #[link_name = "llvm.arm.neon.vshiftins.v2i64"]
    fn vshiftins_v2i64(a: int64x2_t, b: int64x2_t, shift: int64x2_t) -> int64x2_t;

    #[link_name = "llvm.arm.neon.vld1.v8i8.p0i8"]
    fn vld1_v8i8(addr: *const i8, align: i32) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vld1.v16i8.p0i8"]
    fn vld1q_v16i8(addr: *const i8, align: i32) -> int8x16_t;
    #[link_name = "llvm.arm.neon.vld1.v4i16.p0i8"]
    fn vld1_v4i16(addr: *const i8, align: i32) -> int16x4_t;
    #[link_name = "llvm.arm.neon.vld1.v8i16.p0i8"]
    fn vld1q_v8i16(addr: *const i8, align: i32) -> int16x8_t;
    #[link_name = "llvm.arm.neon.vld1.v2i32.p0i8"]
    fn vld1_v2i32(addr: *const i8, align: i32) -> int32x2_t;
    #[link_name = "llvm.arm.neon.vld1.v4i32.p0i8"]
    fn vld1q_v4i32(addr: *const i8, align: i32) -> int32x4_t;
    #[link_name = "llvm.arm.neon.vld1.v1i64.p0i8"]
    fn vld1_v1i64(addr: *const i8, align: i32) -> int64x1_t;
    #[link_name = "llvm.arm.neon.vld1.v2i64.p0i8"]
    fn vld1q_v2i64(addr: *const i8, align: i32) -> int64x2_t;
    #[link_name = "llvm.arm.neon.vld1.v2f32.p0i8"]
    fn vld1_v2f32(addr: *const i8, align: i32) -> float32x2_t;
    #[link_name = "llvm.arm.neon.vld1.v4f32.p0i8"]
    fn vld1q_v4f32(addr: *const i8, align: i32) -> float32x4_t;

    #[link_name = "llvm.arm.neon.vst1.p0i8.v8i8"]
    fn vst1_v8i8(addr: *const i8, val: int8x8_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v16i8"]
    fn vst1q_v16i8(addr: *const i8, val: int8x16_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v4i16"]
    fn vst1_v4i16(addr: *const i8, val: int16x4_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v8i16"]
    fn vst1q_v8i16(addr: *const i8, val: int16x8_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v2i32"]
    fn vst1_v2i32(addr: *const i8, val: int32x2_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v4i32"]
    fn vst1q_v4i32(addr: *const i8, val: int32x4_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v1i64"]
    fn vst1_v1i64(addr: *const i8, val: int64x1_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v2i64"]
    fn vst1q_v2i64(addr: *const i8, val: int64x2_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v2f32"]
    fn vst1_v2f32(addr: *const i8, val: float32x2_t, align: i32);
    #[link_name = "llvm.arm.neon.vst1.p0i8.v4f32"]
    fn vst1q_v4f32(addr: *const i8, val: float32x4_t, align: i32);
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1_s8(ptr: *const i8) -> int8x8_t {
    vld1_v8i8(ptr as *const i8, align_of::<i8>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1q_s8(ptr: *const i8) -> int8x16_t {
    vld1q_v16i8(ptr as *const i8, align_of::<i8>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1_s16(ptr: *const i16) -> int16x4_t {
    vld1_v4i16(ptr as *const i8, align_of::<i16>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1q_s16(ptr: *const i16) -> int16x8_t {
    vld1q_v8i16(ptr as *const i8, align_of::<i16>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vldr))]
pub unsafe fn vld1_s32(ptr: *const i32) -> int32x2_t {
    vld1_v2i32(ptr as *const i8, align_of::<i32>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.32"))]
pub unsafe fn vld1q_s32(ptr: *const i32) -> int32x4_t {
    vld1q_v4i32(ptr as *const i8, align_of::<i32>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vldr))]
pub unsafe fn vld1_s64(ptr: *const i64) -> int64x1_t {
    vld1_v1i64(ptr as *const i8, align_of::<i64>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.64"))]
pub unsafe fn vld1q_s64(ptr: *const i64) -> int64x2_t {
    vld1q_v2i64(ptr as *const i8, align_of::<i64>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1_u8(ptr: *const u8) -> uint8x8_t {
    transmute(vld1_v8i8(ptr as *const i8, align_of::<u8>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1q_u8(ptr: *const u8) -> uint8x16_t {
    transmute(vld1q_v16i8(ptr as *const i8, align_of::<u8>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1_u16(ptr: *const u16) -> uint16x4_t {
    transmute(vld1_v4i16(ptr as *const i8, align_of::<u16>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1q_u16(ptr: *const u16) -> uint16x8_t {
    transmute(vld1q_v8i16(ptr as *const i8, align_of::<u16>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vldr))]
pub unsafe fn vld1_u32(ptr: *const u32) -> uint32x2_t {
    transmute(vld1_v2i32(ptr as *const i8, align_of::<u32>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.32"))]
pub unsafe fn vld1q_u32(ptr: *const u32) -> uint32x4_t {
    transmute(vld1q_v4i32(ptr as *const i8, align_of::<u32>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vldr))]
pub unsafe fn vld1_u64(ptr: *const u64) -> uint64x1_t {
    transmute(vld1_v1i64(ptr as *const i8, align_of::<u64>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.64"))]
pub unsafe fn vld1q_u64(ptr: *const u64) -> uint64x2_t {
    transmute(vld1q_v2i64(ptr as *const i8, align_of::<u64>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1_p8(ptr: *const p8) -> poly8x8_t {
    transmute(vld1_v8i8(ptr as *const i8, align_of::<p8>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.8"))]
pub unsafe fn vld1q_p8(ptr: *const p8) -> poly8x16_t {
    transmute(vld1q_v16i8(ptr as *const i8, align_of::<p8>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1_p16(ptr: *const p16) -> poly16x4_t {
    transmute(vld1_v4i16(ptr as *const i8, align_of::<p16>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.16"))]
pub unsafe fn vld1q_p16(ptr: *const p16) -> poly16x8_t {
    transmute(vld1q_v8i16(ptr as *const i8, align_of::<p16>() as i32))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vldr))]
pub unsafe fn vld1_f32(ptr: *const f32) -> float32x2_t {
    vld1_v2f32(ptr as *const i8, align_of::<f32>() as i32)
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vld1.32"))]
pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t {
    vld1q_v4f32(ptr as *const i8, align_of::<f32>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_s8(ptr: *mut i8, a: int8x8_t) {
    vst1_v8i8(ptr as *const i8, a, align_of::<i8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_s8(ptr: *mut i8, a: int8x16_t) {
    vst1q_v16i8(ptr as *const i8, a, align_of::<i8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_s16(ptr: *mut i16, a: int16x4_t) {
    vst1_v4i16(ptr as *const i8, a, align_of::<i16>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_s16(ptr: *mut i16, a: int16x8_t) {
    vst1q_v8i16(ptr as *const i8, a, align_of::<i16>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_s32(ptr: *mut i32, a: int32x2_t) {
    vst1_v2i32(ptr as *const i8, a, align_of::<i32>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_s32(ptr: *mut i32, a: int32x4_t) {
    vst1q_v4i32(ptr as *const i8, a, align_of::<i32>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_s64(ptr: *mut i64, a: int64x1_t) {
    vst1_v1i64(ptr as *const i8, a, align_of::<i64>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_s64(ptr: *mut i64, a: int64x2_t) {
    vst1q_v2i64(ptr as *const i8, a, align_of::<i64>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_u8(ptr: *mut u8, a: uint8x8_t) {
    vst1_v8i8(ptr as *const i8, transmute(a), align_of::<u8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_u8(ptr: *mut u8, a: uint8x16_t) {
    vst1q_v16i8(ptr as *const i8, transmute(a), align_of::<u8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_u16(ptr: *mut u16, a: uint16x4_t) {
    vst1_v4i16(ptr as *const i8, transmute(a), align_of::<u16>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_u16(ptr: *mut u16, a: uint16x8_t) {
    vst1q_v8i16(ptr as *const i8, transmute(a), align_of::<u16>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_u32(ptr: *mut u32, a: uint32x2_t) {
    vst1_v2i32(ptr as *const i8, transmute(a), align_of::<u32>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_u32(ptr: *mut u32, a: uint32x4_t) {
    vst1q_v4i32(ptr as *const i8, transmute(a), align_of::<u32>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_u64(ptr: *mut u64, a: uint64x1_t) {
    vst1_v1i64(ptr as *const i8, transmute(a), align_of::<u64>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_u64(ptr: *mut u64, a: uint64x2_t) {
    vst1q_v2i64(ptr as *const i8, transmute(a), align_of::<u64>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_p8(ptr: *mut p8, a: poly8x8_t) {
    vst1_v8i8(ptr as *const i8, transmute(a), align_of::<p8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_p8(ptr: *mut p8, a: poly8x16_t) {
    vst1q_v16i8(ptr as *const i8, transmute(a), align_of::<p8>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_p16(ptr: *mut p16, a: poly16x4_t) {
    vst1_v4i16(ptr as *const i8, transmute(a), align_of::<p16>() as i32)
}

/// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_p16(ptr: *mut p16, a: poly16x8_t) {
    vst1q_v8i16(ptr as *const i8, transmute(a), align_of::<p8>() as i32)
}

// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1_f32(ptr: *mut f32, a: float32x2_t) {
    vst1_v2f32(ptr as *const i8, a, align_of::<f32>() as i32)
}

// Store multiple single-element structures from one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vst1q_f32(ptr: *mut f32, a: float32x4_t) {
    vst1q_v4f32(ptr as *const i8, a, align_of::<f32>() as i32)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vtbl1(a, b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl1(transmute(a), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl1_p8(a: poly8x8_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl1(transmute(a), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_s8(a: int8x8x2_t, b: int8x8_t) -> int8x8_t {
    vtbl2(a.0, a.1, b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_u8(a: uint8x8x2_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl2(transmute(a.0), transmute(a.1), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl2_p8(a: poly8x8x2_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl2(transmute(a.0), transmute(a.1), transmute(b)))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_s8(a: int8x8x3_t, b: int8x8_t) -> int8x8_t {
    vtbl3(a.0, a.1, a.2, b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_u8(a: uint8x8x3_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl3(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl3_p8(a: poly8x8x3_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl3(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_s8(a: int8x8x4_t, b: int8x8_t) -> int8x8_t {
    vtbl4(a.0, a.1, a.2, a.3, b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_u8(a: uint8x8x4_t, b: uint8x8_t) -> uint8x8_t {
    transmute(vtbl4(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(a.3),
        transmute(b),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbl))]
pub unsafe fn vtbl4_p8(a: poly8x8x4_t, b: uint8x8_t) -> poly8x8_t {
    transmute(vtbl4(
        transmute(a.0),
        transmute(a.1),
        transmute(a.2),
        transmute(a.3),
        transmute(b),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_s8(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t {
    vtbx1(a, b, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx1(transmute(a), transmute(b), transmute(c)))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx1_p8(a: poly8x8_t, b: poly8x8_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx1(transmute(a), transmute(b), transmute(c)))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_s8(a: int8x8_t, b: int8x8x2_t, c: int8x8_t) -> int8x8_t {
    vtbx2(a, b.0, b.1, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_u8(a: uint8x8_t, b: uint8x8x2_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx2(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx2_p8(a: poly8x8_t, b: poly8x8x2_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx2(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_s8(a: int8x8_t, b: int8x8x3_t, c: int8x8_t) -> int8x8_t {
    vtbx3(a, b.0, b.1, b.2, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_u8(a: uint8x8_t, b: uint8x8x3_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx3(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx3_p8(a: poly8x8_t, b: poly8x8x3_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx3(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_s8(a: int8x8_t, b: int8x8x4_t, c: int8x8_t) -> int8x8_t {
    vtbx4(a, b.0, b.1, b.2, b.3, c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_u8(a: uint8x8_t, b: uint8x8x4_t, c: uint8x8_t) -> uint8x8_t {
    transmute(vtbx4(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(b.3),
        transmute(c),
    ))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr(vtbx))]
pub unsafe fn vtbx4_p8(a: poly8x8_t, b: poly8x8x4_t, c: uint8x8_t) -> poly8x8_t {
    transmute(vtbx4(
        transmute(a),
        transmute(b.0),
        transmute(b.1),
        transmute(b.2),
        transmute(b.3),
        transmute(c),
    ))
}

// These float-to-int implementations have undefined behaviour when `a` overflows
// the destination type. Clang has the same problem: https://llvm.org/PR47510

/// Floating-point Convert to Signed fixed-point, rounding toward Zero (vector)
#[inline]
#[target_feature(enable = "neon")]
#[target_feature(enable = "v7")]
#[cfg_attr(test, assert_instr("vcvt.s32.f32"))]
pub unsafe fn vcvtq_s32_f32(a: float32x4_t) -> int32x4_t {
    transmute(simd_cast::<_, i32x4>(transmute::<_, f32x4>(a)))
}

/// Floating-point Convert to Unsigned fixed-point, rounding toward Zero (vector)
#[inline]
#[target_feature(enable = "neon")]
#[target_feature(enable = "v7")]
#[cfg_attr(test, assert_instr("vcvt.u32.f32"))]
pub unsafe fn vcvtq_u32_f32(a: float32x4_t) -> uint32x4_t {
    transmute(simd_cast::<_, u32x4>(transmute::<_, f32x4>(a)))
}

/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    let n = N as i8;
    vshiftins_v8i8(a, b, int8x8_t(n, n, n, n, n, n, n, n))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert_imm3!(N);
    let n = N as i8;
    vshiftins_v16i8(
        a,
        b,
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    )
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm4!(N);
    let n = N as i16;
    vshiftins_v4i16(a, b, int16x4_t(n, n, n, n))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm4!(N);
    let n = N as i16;
    vshiftins_v8i16(a, b, int16x8_t(n, n, n, n, n, n, n, n))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    vshiftins_v2i32(a, b, int32x2_t(N, N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    vshiftins_v4i32(a, b, int32x4_t(N, N, N, N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where 0 <= N && N <= 63);
    vshiftins_v1i64(a, b, int64x1_t(N as i64))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where 0 <= N && N <= 63);
    vshiftins_v2i64(a, b, int64x2_t(N as i64, N as i64))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    let n = N as i8;
    transmute(vshiftins_v8i8(
        transmute(a),
        transmute(b),
        int8x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert_imm3!(N);
    let n = N as i8;
    transmute(vshiftins_v16i8(
        transmute(a),
        transmute(b),
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm4!(N);
    let n = N as i16;
    transmute(vshiftins_v4i16(
        transmute(a),
        transmute(b),
        int16x4_t(n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm4!(N);
    let n = N as i16;
    transmute(vshiftins_v8i16(
        transmute(a),
        transmute(b),
        int16x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    transmute(vshiftins_v2i32(transmute(a), transmute(b), int32x2_t(N, N)))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    transmute(vshiftins_v4i32(
        transmute(a),
        transmute(b),
        int32x4_t(N, N, N, N),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where 0 <= N && N <= 63);
    transmute(vshiftins_v1i64(
        transmute(a),
        transmute(b),
        int64x1_t(N as i64),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where 0 <= N && N <= 63);
    transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t(N as i64, N as i64),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_p8<const N: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(N);
    let n = N as i8;
    transmute(vshiftins_v8i8(
        transmute(a),
        transmute(b),
        int8x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p8<const N: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert_imm3!(N);
    let n = N as i8;
    transmute(vshiftins_v16i8(
        transmute(a),
        transmute(b),
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_p16<const N: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert_imm4!(N);
    let n = N as i16;
    transmute(vshiftins_v4i16(
        transmute(a),
        transmute(b),
        int16x4_t(n, n, n, n),
    ))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsli.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p16<const N: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert_imm4!(N);
    let n = N as i16;
    transmute(vshiftins_v8i16(
        transmute(a),
        transmute(b),
        int16x8_t(n, n, n, n, n, n, n, n),
    ))
}

/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    vshiftins_v8i8(a, b, int8x8_t(n, n, n, n, n, n, n, n))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    vshiftins_v16i8(
        a,
        b,
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    )
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    vshiftins_v4i16(a, b, int16x4_t(n, n, n, n))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    vshiftins_v8i16(a, b, int16x8_t(n, n, n, n, n, n, n, n))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N : i32 where 1 <= N && N <= 32);
    vshiftins_v2i32(a, b, int32x2_t(-N, -N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N : i32 where 1 <= N && N <= 32);
    vshiftins_v4i32(a, b, int32x4_t(-N, -N, -N, -N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where 1 <= N && N <= 64);
    vshiftins_v1i64(a, b, int64x1_t(-N as i64))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where 1 <= N && N <= 64);
    vshiftins_v2i64(a, b, int64x2_t(-N as i64, -N as i64))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    transmute(vshiftins_v8i8(
        transmute(a),
        transmute(b),
        int8x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    transmute(vshiftins_v16i8(
        transmute(a),
        transmute(b),
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    transmute(vshiftins_v4i16(
        transmute(a),
        transmute(b),
        int16x4_t(n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    transmute(vshiftins_v8i16(
        transmute(a),
        transmute(b),
        int16x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where 1 <= N && N <= 32);
    transmute(vshiftins_v2i32(
        transmute(a),
        transmute(b),
        int32x2_t(-N, -N),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.32", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where 1 <= N && N <= 32);
    transmute(vshiftins_v4i32(
        transmute(a),
        transmute(b),
        int32x4_t(-N, -N, -N, -N),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where 1 <= N && N <= 64);
    transmute(vshiftins_v1i64(
        transmute(a),
        transmute(b),
        int64x1_t(-N as i64),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where 1 <= N && N <= 64);
    transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t(-N as i64, -N as i64),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_p8<const N: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    transmute(vshiftins_v8i8(
        transmute(a),
        transmute(b),
        int8x8_t(n, n, n, n, n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.8", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p8<const N: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert!(N : i32 where 1 <= N && N <= 8);
    let n = -N as i8;
    transmute(vshiftins_v16i8(
        transmute(a),
        transmute(b),
        int8x16_t(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_p16<const N: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    transmute(vshiftins_v4i16(
        transmute(a),
        transmute(b),
        int16x4_t(n, n, n, n),
    ))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(test, assert_instr("vsri.16", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p16<const N: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert!(N : i32 where 1 <= N && N <= 16);
    let n = -N as i16;
    transmute(vshiftins_v8i16(
        transmute(a),
        transmute(b),
        int16x8_t(n, n, n, n, n, n, n, n),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_arch::{arm::*, simd::*};
    use crate::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_s32_f32() {
        let f = f32x4::new(-1., 2., 3., 4.);
        let e = i32x4::new(-1, 2, 3, 4);
        let r: i32x4 = transmute(vcvtq_s32_f32(transmute(f)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_u32_f32() {
        let f = f32x4::new(1., 2., 3., 4.);
        let e = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vcvtq_u32_f32(transmute(f)));
        assert_eq!(r, e);
    }
}
