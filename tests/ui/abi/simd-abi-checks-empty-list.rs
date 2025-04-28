//! At the time of writing, the list of "which target feature enables which vector size" is empty
//! for SPARC. Ensure that this leads to all vector sizes causing an error.
//@ add-core-stubs
//@ needs-llvm-components: sparc
//@ compile-flags: --target=sparc-unknown-none-elf --crate-type=rlib
//@ build-fail
#![no_core]
#![feature(no_core, repr_simd)]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct SimdVec([i32; 4]);

pub extern "C" fn pass_by_vec(_: SimdVec) {}
//~^ ERROR: this function definition uses SIMD vector type `SimdVec` which is not currently supported with the chosen ABI
