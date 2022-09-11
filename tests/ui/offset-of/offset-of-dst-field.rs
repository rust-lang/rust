#![feature(offset_of)]

use std::mem::offset_of;

#[repr(C)]
struct Foo {
    x: u8,
    y: u16,
    slice: [u8],
}

fn main() {
    offset_of!(Foo, slice); //~ ERROR the size for values of type
}
