//@ add-core-stubs
//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_cmse_nonsecure_call, cmse_nonsecure_entry, no_core, lang_items, intrinsics)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn test(
    f: extern "cmse-nonsecure-call" fn(u32, u32, u32, u32) -> u32,
    a: u32,
    b: u32,
    c: u32,
) -> u32 {
    f(a, b, c, 42)
}
