//@ assembly-output: emit-asm
//
//@ revisions: AARCH64
//@[AARCH64] compile-flags: -Copt-level=3
//@[AARCH64] only-aarch64
//
//@ revisions: ARMV7
//@[ARMV7] compile-flags: -Copt-level=3
//@[ARMV7] only-arm
//@[ARMV7] ignore-thumb
//@[ARMV7] ignore-android
#![crate_type = "lib"]
#![cfg_attr(target_arch = "arm", feature(arm_target_feature, stdarch_arm_neon_intrinsics))]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;

// Loads of 3 are error-prone because a `repr(simd)` type's size is always rounded up to the next
// power of 2. Hence, using `read_unaligned` and `write_unaligned` on such types is invalid, it
// would go out of bounds.
#[unsafe(no_mangle)]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon,v7"))]
fn test_vld3q_f32(ptr: *const f32) -> float32x4x3_t {
    // AARCH64-LABEL: test_vld3q_f32
    // AARCH64: ld3 { v0.4s, v1.4s, v2.4s }, [x0]
    // AARCH64: stp q0, q1, [x8]
    // AARCH64: str q2, [x8, #32]
    // AARCH64: ret
    //
    // ARMV7-LABEL: test_vld3q_f32
    // ARMV7: vld3.32 {d16, d18, d20}, [r1]!
    // ARMV7: vld3.32 {d17, d19, d21}, [r1]
    // ARMV7: vst1.32 {d16, d17}, [r0]!
    // ARMV7: vst1.32 {d18, d19}, [r0]!
    // ARMV7: vst1.64 {d20, d21}, [r0]
    // ARMV7: bx lr
    unsafe { vld3q_f32(ptr) }
}

#[unsafe(no_mangle)]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon,v7"))]
fn test_vld3q_s32(ptr: *const i32) -> int32x4x3_t {
    // AARCH64-LABEL: test_vld3q_s32
    // AARCH64: ld3 { v0.4s, v1.4s, v2.4s }, [x0]
    // AARCH64: stp q0, q1, [x8]
    // AARCH64: str q2, [x8, #32]
    // AARCH64: ret
    //
    // ARMV7-LABEL: test_vld3q_s32
    // ARMV7: vld3.32 {d16, d18, d20}, [r1]!
    // ARMV7: vld3.32 {d17, d19, d21}, [r1]
    // ARMV7: vst1.32 {d16, d17}, [r0]!
    // ARMV7: vst1.32 {d18, d19}, [r0]!
    // ARMV7: vst1.64 {d20, d21}, [r0]
    // ARMV7: bx lr
    unsafe { vld3q_s32(ptr) }
}

#[unsafe(no_mangle)]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon,v7"))]
fn test_vld3q_u32(ptr: *const u32) -> uint32x4x3_t {
    // AARCH64-LABEL: test_vld3q_u32
    // AARCH64: ld3 { v0.4s, v1.4s, v2.4s }, [x0]
    // AARCH64: stp q0, q1, [x8]
    // AARCH64: str q2, [x8, #32]
    // AARCH64: ret
    //
    // ARMV7-LABEL: test_vld3q_u32
    // ARMV7: vld3.32 {d16, d18, d20}, [r1]!
    // ARMV7: vld3.32 {d17, d19, d21}, [r1]
    // ARMV7: vst1.32 {d16, d17}, [r0]!
    // ARMV7: vst1.32 {d18, d19}, [r0]!
    // ARMV7: vst1.64 {d20, d21}, [r0]
    // ARMV7: bx lr
    unsafe { vld3q_u32(ptr) }
}
