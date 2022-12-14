// run-pass

#![allow(dead_code)]
#![allow(unused_variables)]
// Checks if the correct registers are being used to pass arguments
// when the sysv64 ABI is specified.

#![feature(rust_2018_preview)]

pub trait Foo {}

// should compile without the dyn trait feature flag
fn foo(x: &dyn Foo) {}

pub fn main() {}
