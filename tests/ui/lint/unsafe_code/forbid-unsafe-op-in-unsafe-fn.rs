#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]
#![allow(dead_code)]
#![deny(unsafe_code)]

struct Bar;

unsafe fn baz() {}

trait Baz {
    unsafe fn baz(&self);
    unsafe fn provided(&self) {}
    unsafe fn provided_override(&self) {}
}

impl Baz for Bar {
    unsafe fn baz(&self) {}
    unsafe fn provided_override(&self) {}
}

unsafe fn unsafe_op_in_unsafe_fn() {
    unsafe {} //~ ERROR: usage of an `unsafe` block
}

fn main() {}
