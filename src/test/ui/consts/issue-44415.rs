#![feature(core_intrinsics)]

use std::intrinsics;

struct Foo {
    bytes: [u8; unsafe { intrinsics::size_of::<Foo>() }],
    //~^ ERROR cycle detected when simplifying constant for the type system
    x: usize,
}

fn main() {}
