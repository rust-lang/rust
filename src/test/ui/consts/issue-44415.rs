#![feature(core_intrinsics)]

use std::intrinsics;

struct Foo {
    bytes: [u8; unsafe { intrinsics::size_of::<Foo>() }],
    //~^ ERROR cycle detected when const-evaluating + checking `Foo::bytes::{constant#0}`
    x: usize,
}

fn main() {}
