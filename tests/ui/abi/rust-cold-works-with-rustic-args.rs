//@ add-core-stubs
//@ build-pass
//@ compile-flags: -Clink-dead-code=true
// We used to not handle all "rustic" ABIs in a (relatively) uniform way,
// so we failed to fix up arguments for actually passing through the ABI...
#![feature(rust_cold_cc)]
#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

pub extern "rust-cold" fn foo(_: [usize; 3]) {}
