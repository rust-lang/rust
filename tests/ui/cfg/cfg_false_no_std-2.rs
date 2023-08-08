// Error, the linked empty library is `no_std` and doesn't provide a panic handler.

// dont-check-compiler-stderr
// error-pattern: `#[panic_handler]` function required, but not found
// aux-build: cfg_false_lib_no_std_before.rs

#![no_std]

extern crate cfg_false_lib_no_std_before as _;

fn main() {}
