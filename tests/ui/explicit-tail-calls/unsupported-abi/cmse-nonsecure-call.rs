//@ add-minicore
//@ ignore-backends: gcc
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![expect(incomplete_features)]
#![feature(no_core, explicit_tail_calls, abi_cmse_nonsecure_call)]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe extern "C" {
    safe fn magic() -> extern "cmse-nonsecure-call" fn(u32, u32) -> u32;
}

// The `cmse-nonsecure-call` ABI can only occur on function pointers:
//
// - a `cmse-nonsecure-call` definition throws an error
// - a `cmse-nonsecure-call` become in a definition with any other ABI is an ABI mismatch
#[no_mangle]
extern "cmse-nonsecure-call" fn become_nonsecure_call_1(x: u32, y: u32) -> u32 {
    //~^ ERROR the `"cmse-nonsecure-call"` ABI is only allowed on function pointers
    unsafe {
        let f = magic();
        become f(1, 2)
        //~^ ERROR ABI does not support guaranteed tail calls
    }
}

#[no_mangle]
extern "C" fn become_nonsecure_call_2(x: u32, y: u32) -> u32 {
    unsafe {
        let f = magic();
        become f(1, 2)
        //~^ ERROR mismatched function ABIs
        //~| ERROR ABI does not support guaranteed tail calls
    }
}
