//! At the time of writing, the list of "which target feature enables which vector size" is empty
//! for SPARC. Ensure that this leads to all vector sizes causing an error.
//@ add-minicore
//@ needs-llvm-components: sparc
//@ compile-flags: --target=sparc-unknown-none-elf --crate-type=rlib
//@ build-fail
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core)]

extern crate minicore;
use minicore::simd::Simd;

#[expect(improper_ctypes_definitions)]
pub extern "C" fn pass_by_vec(_: Simd<i32, 4>) {}
//~^ ERROR: this function definition uses SIMD vector type `Simd<i32, 4>` which is not currently supported with the chosen ABI
