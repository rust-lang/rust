//@ add-minicore
//@ ignore-backends: gcc
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![expect(incomplete_features)]
#![feature(no_core, explicit_tail_calls, cmse_nonsecure_entry)]
#![no_core]

extern crate minicore;
use minicore::*;

#[inline(never)]
extern "cmse-nonsecure-entry" fn entry(c: bool, x: u32, y: u32) -> u32 {
    if c { x } else { y }
}

// A `cmse-nonsecure-entry` clears registers before returning, so a tail call cannot be guaranteed.
#[unsafe(no_mangle)]
extern "cmse-nonsecure-entry" fn become_nonsecure_entry(c: bool, x: u32, y: u32) -> u32 {
    become entry(c, x, y)
    //~^ ERROR ABI does not support guaranteed tail calls
}
