// unit-test
// compile-flags: -O

// EMIT_MIR offset_of.main.ConstProp.diff

#![feature(offset_of)]

use std::mem::offset_of;

#[repr(C)]
struct Foo {
    x: u8,
    y: u16,
    z: Bar,
}

#[repr(C)]
struct Bar(u8, u8);

fn main() {
    let x = offset_of!(Foo, x);
    let y = offset_of!(Foo, y);
    let z0 = offset_of!(Foo, z.0);
    let z1 = offset_of!(Foo, z.1);
}
