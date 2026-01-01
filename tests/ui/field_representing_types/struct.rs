//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

#[repr(C)]
pub struct Foo {
    a: i32,
    b: i64,
}

fn main() {
    assert_eq!(<field_of!(Foo, a)>::OFFSET, 0);
    assert_eq!(<field_of!(Foo, b)>::OFFSET, 8);
}
