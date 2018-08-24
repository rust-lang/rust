//~^^^^^^^^^^ ERROR cycle detected when computing layout of

#![feature(const_fn)]
#![feature(core_intrinsics)]

use std::intrinsics;

struct Foo {
    bytes: [u8; unsafe { intrinsics::size_of::<Foo>() }],
    x: usize,
}

fn main() {}
