//@ add-minicore
//@ build-fail
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![feature(no_core, link_llvm_intrinsics, simd_ffi)]
#![no_core]
#![allow(internal_features, non_camel_case_types)]
#![crate_type = "lib"]

extern crate minicore;
use minicore::simd::i8x8;
use minicore::*;

extern "llvm-intrinsic" {
    #[deny(deprecated_llvm_intrinsic)]
    #[link_name = "llvm.aarch64.neon.rbit.v8i8"]
    fn rbit(a: i8x8) -> i8x8;
    //~^ ERROR: using deprecated intrinsic `llvm.aarch64.neon.rbit.v8i8`, `llvm.bitreverse.v8i8` can be used instead

    #[deny(deprecated_llvm_intrinsic)]
    #[link_name = "llvm.aarch64.neon.vcvthf2fp"]
    fn vcvthf2fp(a: u32) -> f32;
    //~^ ERROR: using deprecated intrinsic `llvm.aarch64.neon.vcvthf2fp`
}

#[target_feature(enable = "neon")]
pub unsafe fn bar(a: i8x8, b: u32) -> (i8x8, f32) {
    (rbit(a), vcvthf2fp(b))
}
