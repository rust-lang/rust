// Error, the linked empty library is `no_std` and doesn't provide a panic handler.

//@ dont-check-compiler-stderr

// NOTE: fix a panic strategy to prevent differing errors subject to target's default panic strategy
// which changes between targets. The specific panic strategy doesn't matter for test intention.
//@ compile-flags: -Cpanic=abort

//@ aux-build: cfg_false_lib_no_std_before.rs

#![no_std]

extern crate cfg_false_lib_no_std_before as _;

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
