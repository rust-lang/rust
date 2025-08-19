//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C, align(16))]
#[allow(unused)]
pub struct AlignRelevant(u32);

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f1(_: u32, _: u32, _: u32, _: u32, _: u32, _: u32) {} //~ ERROR [E0798]
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f2(_: u32, _: u32, _: u32, _: u16, _: u16) {} //~ ERROR [E0798]
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f3(_: u32, _: u64, _: u32) {} //~ ERROR [E0798]
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn f4(_: AlignRelevant, _: u32) {} //~ ERROR [E0798]

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn f5(_: [u32; 5]) {} //~ ERROR [E0798]
