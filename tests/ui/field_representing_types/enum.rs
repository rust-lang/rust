//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

#[repr(C, i8)]
pub enum Foo {
    A { a: i32, b: i64 },
    B { x: i64, y: i32 },
}

fn main() {
    assert_eq!(<field_of!(Foo, A.a)>::OFFSET, 8);
    assert_eq!(<field_of!(Foo, A.b)>::OFFSET, 16);

    assert_eq!(<field_of!(Foo, B.x)>::OFFSET, 8);
    assert_eq!(<field_of!(Foo, B.y)>::OFFSET, 16);
}
