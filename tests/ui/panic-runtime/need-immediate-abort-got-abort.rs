//@ build-fail
//@ aux-build:needs-immediate-abort.rs
//@ compile-flags:-C panic=abort
//@ no-prefer-dynamic
//@ add-core-stubs
//@ core-stubs-compile-flags: -Zunstable-options -Cpanic=immediate-abort

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]

extern crate minicore;
extern crate needs_immediate_abort;

extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    0
}

//~? ERROR the crate `need_immediate_abort_got_abort` was compiled with a panic strategy which is incompatible with `immediate-abort`
