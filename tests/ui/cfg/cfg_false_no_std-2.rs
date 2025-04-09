// Error, the linked empty library is `no_std` and doesn't provide a panic handler.

//@ dont-require-annotations:ERROR
//@ dont-check-compiler-stderr
//@ aux-build: cfg_false_lib_no_std_before.rs

#![no_std]

extern crate cfg_false_lib_no_std_before as _;

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
// FIXME: This error is target-dependent, could be served by some "optional error" annotation
// instead of `dont-require-annotations`.
//FIXME~? ERROR unwinding panics are not supported without std
