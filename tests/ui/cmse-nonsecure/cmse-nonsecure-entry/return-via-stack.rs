//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm

#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]
#![warn(cmse_uninitialized_leak)]

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
pub extern "cmse-nonsecure-entry" fn f1() -> ReprCU64 {
    //~^ ERROR [E0798]
    ReprCU64(0)
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f2() -> ReprCBytes {
    //~^ ERROR [E0798]
    ReprCBytes(0, 1, 2, 3, 4)
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f3() -> U64Compound {
    //~^ ERROR [E0798]
    U64Compound(2, 3)
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f4() -> ReprCAlign16 {
    //~^ ERROR [E0798]
    ReprCAlign16(4)
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn f5() -> [u8; 5] {
    //~^ ERROR [E0798]
    [0xAA; 5]
}
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn u128() -> u128 {
    //~^ ERROR [E0798]
    123
}
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn i128() -> i128 {
    //~^ ERROR [E0798]
    456
}
