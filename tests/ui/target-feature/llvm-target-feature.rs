//! Ensure `-Zllvm-target-feature` is honored by `cfg(target_feature)` and by the `extern "C"`
//! vector ABI checks.

//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Zllvm-target-feature=+avx2
//@ needs-llvm-components: x86

//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core, intrinsics, rustc_attrs)]
#![no_core]

extern crate minicore;
use minicore::*;

// Also test the ABI checks by using `extern "C"` (this would error if the feature did not actually
// get enabled).
#[no_mangle] // force codegen
pub extern "C" fn do_thing(x: simd::f32x8, y: simd::f32x8) -> simd::f32x8 {
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub const unsafe fn simd_add<T>(x: T, y: T) -> T;

    unsafe { simd_add(x, y) }
}

#[cfg(not(target_feature = "avx2"))]
compile_error!("the avx2 cfg did not get set");
