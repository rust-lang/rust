//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-aarch64-unknown-linux-gnu
#![feature(repr_simd, portable_simd, core_intrinsics, f16, f128)]
#![crate_type = "lib"]
#![allow(non_camel_case_types)]

// Test `vld_s16` can be implemented in a portable way (i.e. without using LLVM neon intrinsics).
// This relies on rust preserving the SIMD vector element type and using it to construct the
// LLVM type. Without this information, additional casts are needed that defeat the LLVM pattern
// matcher, see https://github.com/llvm/llvm-project/issues/181514.

use std::mem::transmute;
use std::simd::Simd;

#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
unsafe extern "C" fn vld2_s16_old(ptr: *const i16) -> std::arch::aarch64::int16x4x2_t {
    // CHECK-LABEL: vld2_s16_old
    // CHECK: .cfi_startproc
    // CHECK-NEXT: ld2 { v0.4h, v1.4h }, [x0]
    // CHECK-NEXT: ret
    std::arch::aarch64::vld2_s16(ptr)
}

#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
unsafe extern "C" fn vld2_s16_new(a: *const i16) -> std::arch::aarch64::int16x4x2_t {
    // CHECK-LABEL: vld2_s16_new
    // CHECK: .cfi_startproc
    // CHECK-NEXT: ld2 { v0.4h, v1.4h }, [x0]
    // CHECK-NEXT: ret

    type V = Simd<i16, 4>;
    type W = Simd<i16, 8>;

    let w: W = std::ptr::read_unaligned(a as *const W);

    #[repr(simd)]
    pub(crate) struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

    let v0: V = std::intrinsics::simd::simd_shuffle(w, w, const { SimdShuffleIdx([0, 2, 4, 6]) });
    let v1: V = std::intrinsics::simd::simd_shuffle(w, w, const { SimdShuffleIdx([1, 3, 5, 7]) });

    transmute((v0, v1))
}
