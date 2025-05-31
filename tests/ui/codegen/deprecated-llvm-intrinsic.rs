//@ add-core-stubs
//@ build-pass
//@ ignore-pass
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
#![feature(no_core, lang_items, link_llvm_intrinsics, abi_unadjusted, repr_simd, simd_ffi)]
#![no_std]
#![no_core]
#![allow(internal_features, non_camel_case_types, improper_ctypes)]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i8x8([i8; 8]);

extern "unadjusted" {
    #[link_name = "llvm.aarch64.neon.rbit.v8i8"]
    fn foo(a: i8x8) -> i8x8;
}

#[target_feature(enable = "neon")]
pub unsafe fn bar(a: i8x8) -> i8x8 {
    foo(a)
}

//~? NOTE: Using deprecated intrinsic `llvm.aarch64.neon.rbit.v8i8`, `llvm.bitreverse.v8i8` can be used instead
