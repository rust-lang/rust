//@ build-fail
//@ aux-build:needs-unwind-immediate-abort.rs
//@ compile-flags:-C panic=immediate-abort -Zunstable-options
//@ no-prefer-dynamic
//@ add-core-stubs
//@ core-stubs-compile-flags: -Zunstable-options -Cpanic=immediate-abort

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]

extern crate minicore;
extern crate needs_unwind_immediate_abort;

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    0
}

//~? ERROR the crate `needs_unwind_immediate_abort` was compiled with a panic strategy which is incompatible with `immediate-abort`
