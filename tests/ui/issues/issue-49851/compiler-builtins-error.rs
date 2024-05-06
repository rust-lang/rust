//~ ERROR can't find crate for `core`

//@ compile-flags: --target thumbv7em-none-eabihf
//@ needs-llvm-components: arm
#![deny(unsafe_code)]
#![deny(warnings)]
#![no_std]

extern crate cortex_m;

fn main() {}
