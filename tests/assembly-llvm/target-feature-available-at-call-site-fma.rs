//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel -Ctarget-feature=-avx,-fma
//@ only-x86_64
//@ ignore-sgx
//@ ignore-backends: gcc

#![feature(core_intrinsics)]

use std::arch::x86_64::{_mm_add_ss, _mm_cvtss_f32, _mm_fmadd_ss, _mm_mul_ss, _mm_set_ss};
use std::intrinsics::simd::target_feature_available_at_call_site;

#[inline(always)]
fn maybe_fma(x: f32, y: f32, z: f32) -> f32 {
    if target_feature_available_at_call_site!("fma") { x.mul_add(y, z) } else { x * y + z }
}

#[inline(always)]
fn maybe_fma_arch_intrinsic(x: f32, y: f32, z: f32) -> f32 {
    unsafe {
        let x = _mm_set_ss(x);
        let y = _mm_set_ss(y);
        let z = _mm_set_ss(z);
        let result = if target_feature_available_at_call_site!("fma") {
            _mm_fmadd_ss(x, y, z)
        } else {
            _mm_add_ss(_mm_mul_ss(x, y), z)
        };
        _mm_cvtss_f32(result)
    }
}

// CHECK-LABEL: with_fma:
// CHECK: vfmadd
// CHECK-NOT: mulss
// CHECK-NOT: addss
#[no_mangle]
#[target_feature(enable = "avx,fma")]
pub fn with_fma(x: f32, y: f32, z: f32) -> f32 {
    maybe_fma(x, y, z)
}

// CHECK-LABEL: without_fma:
// CHECK: mulss
// CHECK: addss
// CHECK-NOT: vfmadd
#[no_mangle]
pub fn without_fma(x: f32, y: f32, z: f32) -> f32 {
    maybe_fma(x, y, z)
}

#[no_mangle]
#[target_feature(enable = "avx,fma")]
pub fn with_fma_arch_intrinsic(x: f32, y: f32, z: f32) -> f32 {
    maybe_fma_arch_intrinsic(x, y, z)
}

#[no_mangle]
pub fn without_fma_arch_intrinsic(x: f32, y: f32, z: f32) -> f32 {
    maybe_fma_arch_intrinsic(x, y, z)
}

// CHECK: with_fma_arch_intrinsic = with_fma
// CHECK: without_fma_arch_intrinsic = without_fma
