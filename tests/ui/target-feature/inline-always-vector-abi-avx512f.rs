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
fn callee_missing_avx512f(x: &__m256, y: bool) {
    if y {
        callee_missing_avx512f(x, y);
    } else {
        sink(x);
    }
}

// `avx512f` only changes the `__m256` ABI because it implicitly enables `avx`.
#[target_feature(enable = "avx512f")]
fn caller_has_avx512f_abi_mismatch(x: &__m256, y: bool) {
    unsafe { callee_missing_avx512f(x, y) }
    //~^ WARNING call to `#[inline(always)]`-annotated `callee_missing_avx512f` requires the same target features to be inlined [inline_always_mismatching_target_features]
}
