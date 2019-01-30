// run-pass
#![allow(dead_code)]
#![feature(repr_align_enum)]

use std::mem;

// Raising alignment
#[repr(align(8))]
enum Align8 {
    Foo { foo: u32 },
    Bar { bar: u32 },
}

fn main() {
    assert_eq!(mem::align_of::<Align8>(), 8);
    assert_eq!(mem::size_of::<Align8>(), 8);
}
