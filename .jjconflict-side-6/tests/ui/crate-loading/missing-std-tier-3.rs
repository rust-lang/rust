// Ensure that we do not recommend rustup installing tier 3 targets.

//@ compile-flags: --target m68k-unknown-linux-gnu
//@ needs-llvm-components: m68k
//@ rustc-env:CARGO_CRATE_NAME=foo
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]
extern crate core;
//~^ ERROR can't find crate for `core`
//~| NOTE can't find crate
//~| NOTE target may not be installed
//~| HELP consider building the standard library from source with `cargo build -Zbuild-std`
fn main() {}
