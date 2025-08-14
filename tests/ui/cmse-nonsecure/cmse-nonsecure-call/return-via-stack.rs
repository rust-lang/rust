//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ add-core-stubs

#![feature(abi_cmse_nonsecure_call, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct ReprCU64(u64);

#[repr(C)]
pub struct ReprCBytes(u8, u8, u8, u8, u8);

#[repr(C)]
pub struct U64Compound(u32, u32);

#[repr(C, align(16))]
pub struct ReprCAlign16(u16);

#[no_mangle]
pub fn test(
    f1: extern "cmse-nonsecure-call" fn() -> ReprCU64, //~ ERROR [E0798]
    f2: extern "cmse-nonsecure-call" fn() -> ReprCBytes, //~ ERROR [E0798]
    f3: extern "cmse-nonsecure-call" fn() -> U64Compound, //~ ERROR [E0798]
    f4: extern "cmse-nonsecure-call" fn() -> ReprCAlign16, //~ ERROR [E0798]
    f5: extern "cmse-nonsecure-call" fn() -> [u8; 5],  //~ ERROR [E0798]
) {
}

#[allow(improper_ctypes_definitions)]
struct Test {
    u128: extern "cmse-nonsecure-call" fn() -> u128, //~ ERROR [E0798]
    i128: extern "cmse-nonsecure-call" fn() -> i128, //~ ERROR [E0798]
}

#[repr(C)]
pub union ReprCUnionU64 {
    _unused: u64,
}

#[repr(Rust)]
pub union ReprRustUnionU64 {
    _unused: u64,
}

#[no_mangle]
pub fn test_union(
    f1: extern "cmse-nonsecure-call" fn() -> ReprRustUnionU64, //~ ERROR [E0798]
    f2: extern "cmse-nonsecure-call" fn() -> ReprCUnionU64,    //~ ERROR [E0798]
) {
}
