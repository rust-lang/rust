//! Sometimes `-Ctarget-cpu` can *disable* target features that would by default be enabled on the
//! current target. Ensure that we catch the case where those target features are important for the
//! ABI.

//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Tllvm-target-feature=+avx2 -Zunstable-options
//@ needs-llvm-components: x86

//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core, intrinsics, rustc_attrs)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

// Also test the ABI checks by using `extern "C"`
#[no_mangle] // force codegen
pub extern "C" fn do_thing(x: simd::f32x8, y: simd::f32x8) -> simd::f32x8 {
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub const unsafe fn simd_add<T>(x: T, y: T) -> T;

    unsafe { simd_add(x, y) }
}

#[cfg(not(target_feature = "avx2"))]
compile_error!("the avx2 cfg did not get set");
