//~ ERROR 1:1: 1:1: can't find crate for `core` [E0463]

// compile-flags: --target thumbv7em-none-eabihf
#![deny(unsafe_code)]
#![deny(warnings)]
#![no_std]

extern crate cortex_m;
