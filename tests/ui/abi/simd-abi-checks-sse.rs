//! Ensure we trigger abi_unsupported_vector_types for target features that are usually enabled
//! on a target via the base CPU, but disabled in this file via a `-C` flag.
//@ compile-flags: --crate-type=rlib --target=i586-unknown-linux-gnu
//@ compile-flags: -Ctarget-cpu=pentium4 -C target-feature=-sse,-sse2
//@ add-minicore
//@ build-fail
//@ needs-llvm-components: x86
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::simd::Simd;

#[no_mangle]
pub unsafe extern "C" fn f(_: Simd<i64, 2>) {
    //~^ ERROR: this function definition uses SIMD vector type `Simd<i64, 2>` which (with the chosen ABI) requires the `sse` target feature, which is not enabled
}
