//! Ensure we trigger abi_unsupported_vector_types for target features that are usually enabled
//! on a target via the base CPU, but disabled in this file via a `-C` flag.
//@ compile-flags: --crate-type=rlib --target=i586-unknown-linux-gnu
//@ compile-flags: -Ctarget-cpu=pentium4 -C target-feature=-sse,-sse2
//@ add-core-stubs
//@ build-fail
//@ needs-llvm-components: x86
#![feature(no_core, repr_simd)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct SseVector([i64; 2]);

#[no_mangle]
pub unsafe extern "C" fn f(_: SseVector) {
    //~^ ERROR: this function definition uses SIMD vector type `SseVector` which (with the chosen ABI) requires the `sse` target feature, which is not enabled
}
