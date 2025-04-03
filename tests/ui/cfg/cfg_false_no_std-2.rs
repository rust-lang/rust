// Error, the linked empty library is `no_std` and doesn't provide a panic handler.

//@ compile-flags: --error-format=human
//@ error-pattern: `#[panic_handler]` function required, but not found
//@ dont-check-compiler-stderr
//@ aux-build: cfg_false_lib_no_std_before.rs

#![no_std]

extern crate cfg_false_lib_no_std_before as _;

fn main() {}

// FIXME: The second error is target-dependent.
//FIXME~? ERROR `#[panic_handler]` function required, but not found
//FIXME~? ERROR unwinding panics are not supported without std
