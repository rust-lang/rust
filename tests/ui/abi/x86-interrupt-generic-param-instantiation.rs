//! Tests x86 interrupt ABI parameter validation for generic instantiations.
//@ add-minicore
//@ revisions: x64 i686
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ build-fail
//@ ignore-backends: gcc

#![no_core]
#![feature(no_core, abi_x86_interrupt)]

#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct Frame {
    ip: u64,
    cs: u64,
    flags: u64,
    sp: u64,
    ss: u64
}

extern "x86-interrupt" fn handler<F>(_: F) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
//~| NOTE invalid x86-interrupt parameter type: `()`
//~| NOTE functions with the "x86-interrupt" ABI must not use zero-sized types in their signature
//~| ERROR invalid signature for `extern "x86-interrupt"` function
//~| NOTE invalid x86-interrupt frame parameter type: `bool`
//~| NOTE functions with the "x86-interrupt" ABI must have a non-pointer frame parameter that is valid for any bit pattern
extern "x86-interrupt" fn handler_ec<F, E>(_: F, _: E) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
//~| NOTE invalid x86-interrupt parameter type: `()`
//~| NOTE functions with the "x86-interrupt" ABI must not use zero-sized types in their signature
//~| ERROR invalid signature for `extern "x86-interrupt"` function
//~| NOTE invalid x86-interrupt error code parameter type: `bool`
//~| NOTE functions with the "x86-interrupt" ABI must have a machine-word sized integer error code parameter
//~| ERROR invalid signature for `extern "x86-interrupt"` function
//~| NOTE invalid x86-interrupt error code parameter type: `u8`
//~| NOTE functions with the "x86-interrupt" ABI must have a machine-word sized integer error code parameter

pub fn tests() {
    let test_generic_frame_works:  extern "x86-interrupt" fn(Frame) = handler::<Frame>;
    let test_code_works: extern "x86-interrupt" fn(Frame, usize) = handler_ec::<Frame, usize>;
    let test_zst_frame_fails: extern "x86-interrupt" fn(()) = handler::<()>;
    //~^ NOTE the above error was encountered while instantiating `fn handler::<()>`
    let test_zst_code_fails: extern "x86-interrupt" fn(Frame, ()) = handler_ec::<Frame, ()>;
    //~^ NOTE the above error was encountered while instantiating `fn handler_ec::<Frame, ()>`
    let test_niche_frame_fails: extern "x86-interrupt" fn(bool) = handler::<bool>;
    //~^ NOTE the above error was encountered while instantiating `fn handler::<bool>`
    let test_niche_code_fails: extern "x86-interrupt" fn(Frame, bool) = handler_ec::<Frame, bool>;
    //~^ NOTE the above error was encountered while instantiating `fn handler_ec::<Frame, bool>`
    let test_invalid_code_fails: extern "x86-interrupt" fn(Frame, u8) = handler_ec::<Frame, u8>;
    //~^ NOTE the above error was encountered while instantiating `fn handler_ec::<Frame, u8>`
}
