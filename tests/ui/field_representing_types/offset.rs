//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

#[repr(C)]
pub struct Struct {
    a: i32,
    b: i64,
}

// FIXME(FRTs): need to mark these fields as used by the `field_of!` macro.
#[expect(dead_code)]
pub union Union {
    a: i32,
    b: i64,
}

#[repr(C, i8)]
pub enum Enum {
    A { a: i32, b: i64 },
    B { x: i64, y: i32 },
}

fn main() {
    assert_eq!(<field_of!(Struct, a)>::OFFSET, 0);
    assert_eq!(<field_of!(Struct, b)>::OFFSET, 8);

    assert_eq!(<field_of!(Union, a)>::OFFSET, 0);
    assert_eq!(<field_of!(Union, b)>::OFFSET, 0);

    assert_eq!(<field_of!(Enum, A.a)>::OFFSET, 8);
    assert_eq!(<field_of!(Enum, A.b)>::OFFSET, 16);
    assert_eq!(<field_of!(Enum, B.x)>::OFFSET, 8);
    assert_eq!(<field_of!(Enum, B.y)>::OFFSET, 16);
}
