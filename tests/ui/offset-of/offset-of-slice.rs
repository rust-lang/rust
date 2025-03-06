//@run-pass
#![feature(offset_of_slice)]

use std::mem::offset_of;

#[repr(C)]
struct S {
    a: u8,
    b: (u8, u8),
    c: [i32],
}

#[repr(C)]
struct T {
    x: i8,
    y: S,
}

fn main() {
    assert_eq!(offset_of!(S, c), 4);
    assert_eq!(offset_of!(T, y), 4);
    assert_eq!(offset_of!(T, y.c), 8);
}
