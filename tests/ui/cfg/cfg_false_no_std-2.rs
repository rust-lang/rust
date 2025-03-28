// Error, the linked empty library is `no_std` and doesn't provide a panic handler.

//@ dont-check-compiler-stderr
//@ aux-build: cfg_false_lib_no_std_before.rs

#![no_std]

extern crate cfg_false_lib_no_std_before as _;

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
//~? ERROR unwinding panics are not supported without std
