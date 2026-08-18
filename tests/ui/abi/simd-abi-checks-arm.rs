//! Ensure we trigger abi_unsupported_vector_types for target features that are usually enabled
//! on a target via the base CPU, but disabled in this file via a `-C` flag.
//@ compile-flags: --crate-type=rlib --target=armv7-unknown-linux-gnueabihf
//@ add-minicore
//@ build-fail
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
#![feature(no_core, arm_target_feature)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::simd::Simd;

#[no_mangle]
pub unsafe extern "C" fn f(_: Simd<i32, 4>) {
    //~^ ERROR: this function definition uses SIMD vector type `Simd<i32, 4>` which (with the chosen ABI) requires the `neon` target feature, which is not enabled
}

#[no_mangle]
#[target_feature(enable = "neon")]
pub unsafe extern "C" fn neon(_: Simd<i32, 4>) {}

#[no_mangle]
#[target_feature(enable = "mve")]
pub unsafe extern "C" fn mve(_: Simd<i32, 4>) {}
