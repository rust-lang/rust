//@ compile-flags: -O
//@ assembly-output: emit-asm
//@ only-x86_64

#![crate_type = "lib"]
#![feature(struct_target_features)]

// Check that a struct_target_features type causes the compiler to effectively inline intrinsics.

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
struct Avx {}

#[target_feature(enable = "fma")]
struct Fma {}

#[target_feature(from_args)]
pub fn add_simple(_: Avx, v: __m256) -> __m256 {
    // CHECK-NOT: call
    // CHECK: vaddps
    unsafe { _mm256_add_ps(v, v) }
}

// Test that the features don't get inherited from the arguments without the attribute.
pub fn add_simple_noattr(_: Avx, v: __m256) -> __m256 {
    // CHECK: call
    unsafe { _mm256_add_ps(v, v) }
}

#[target_feature(from_args)]
pub fn add_fma_combined(_: &Avx, _: &Fma, v: __m256) -> (__m256, __m256) {
    // CHECK-NOT: call
    // CHECK-DAG: vaddps
    let r1 = unsafe { _mm256_add_ps(v, v) };
    // CHECK-DAG: vfmadd213ps
    let r2 = unsafe { _mm256_fmadd_ps(v, v, v) };
    (r1, r2)
}

#[target_feature(from_args)]
fn add_generic<S>(_: S, v: __m256) -> __m256 {
    // CHECK-NOT: call
    // CHECK: vaddps
    unsafe { _mm256_add_ps(v, v) }
}

pub fn add_using_generic(v: __m256) -> __m256 {
    assert!(is_x86_feature_detected!("avx"));
    add_generic(unsafe { Avx {} }, v)
}
