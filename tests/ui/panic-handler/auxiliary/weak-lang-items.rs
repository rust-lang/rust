//@ no-prefer-dynamic

// This aux-file will require the eh_personality function to be codegen'd, but
// it hasn't been defined just yet. Make sure we don't explode.

#![no_std]
#![crate_type = "rlib"]

struct A;

impl core::ops::Drop for A {
    fn drop(&mut self) {}
}

pub fn foo() {
    let _a = A;
    panic!("wut");
}

mod std {
    pub use core::{option, fmt};
}
