//! Regression test for <https://github.com/rust-lang/rust/issues/37131>.
//! Tests that compiling for a target which is not installed will result in a helpful
//! error message.
//~^^^ ERROR can't find crate for `std`
//~| NOTE target may not be installed
//~| NOTE can't find crate

//@ compile-flags: --target=thumbv6m-none-eabi
//@ ignore-arm
//@ needs-llvm-components: arm
//@ ignore-backends: gcc

fn main() { }
