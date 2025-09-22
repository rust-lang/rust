//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-fail
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

pub enum Foo {
    A { a: isize, b: usize },
}

fn main() {
    assert_eq!(<field_of!(Foo, A.a)>::OFFSET, <field_of!(Foo, A.b)>::OFFSET);
}
