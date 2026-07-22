//! Verify that the wasm `externref` lang type produces real wasm reference
//! types in function signatures.

//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;

#[lang = "externref"]
#[non_exhaustive]
pub struct externref;

// CHECK: .functype describe (externref) -> (externref)
#[no_mangle]
pub extern "C" fn describe(v: externref) -> externref {
    v
}
