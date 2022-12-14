#![feature(core_intrinsics)]

use std::intrinsics;

struct Foo {
    bytes: [u8; unsafe { intrinsics::size_of::<Foo>() }],
    //~^ ERROR cycle detected when evaluating type-level constant
    x: usize,
}

fn main() {}
