//! Regression test for <https://github.com/rust-lang/rust/issues/49851>.
//! Test compiling for a target which is not installed with `no_std`
//! results in a helpful error message.
//~^^^ ERROR can't find crate for `core`

//@ compile-flags: --target thumbv7em-none-eabihf
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
#![deny(unsafe_code)]
#![deny(warnings)]
#![no_std]

extern crate cortex_m;

fn main() {}
