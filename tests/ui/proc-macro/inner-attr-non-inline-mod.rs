//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[deny(unused_attributes)]
mod module_with_attrs;
//~^ ERROR non-inline modules in proc macro input are unstable
//~| ERROR custom inner attributes are unstable

fn main() {}

//~? ERROR custom inner attributes are unstable
//~? ERROR inner macro attributes are unstable
