//@ build-pass
//@ compile-flags: --crate-type=lib --target=x86_64-unknown-linux-gnu
//@ only-x86_64
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(target_feature_inline_always)]
#![allow(dead_code, unused_unsafe)]

use std::arch::x86_64::__m256;

#[inline(never)]
#[target_feature(enable = "sse")]
fn sink(_x: &__m256) {}

#[inline(always)]
#[target_feature(enable = "sse")]
fn callee_missing_avx(x: &__m256, y: bool) {
    if y {
        callee_missing_avx(x, y);
    } else {
        sink(x);
    }
}

#[target_feature(enable = "avx")]
fn caller_has_abi_mismatch(x: &__m256, y: bool) {
    unsafe { callee_missing_avx(x, y) }
    //~^ WARNING call to `#[inline(always)]`-annotated `callee_missing_avx` requires the same target features to be inlined [inline_always_mismatching_target_features]
}

#[inline(always)]
#[target_feature(enable = "avx")]
fn callee_requires_avx(x: &__m256, y: bool) {
    if y {
        callee_requires_avx(x, y);
    } else {
        sink(x);
    }
}

#[target_feature(enable = "sse")]
fn caller_missing_avx(x: &__m256, y: bool) {
    unsafe { callee_requires_avx(x, y) }
    //~^ WARNING call to `#[inline(always)]`-annotated `callee_requires_avx` requires the same target features to be inlined [inline_always_mismatching_target_features]
}
