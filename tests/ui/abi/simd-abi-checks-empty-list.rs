//@ add-core-stubs
//@ needs-llvm-components: sparc
//@ compile-flags: --target=sparc-unknown-none-elf --crate-type=rlib
//@ build-pass
//@ ignore-pass (test emits codegen-time warnings)
#![no_core]
#![feature(no_core, repr_simd)]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct SimdVec([i32; 4]);

pub extern "C" fn pass_by_vec(_: SimdVec) {}
//~^ WARN this function definition uses SIMD vector type `SimdVec` which is not currently supported with the chosen ABI
//~| WARNING this was previously accepted by the compiler
