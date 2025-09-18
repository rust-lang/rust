//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![allow(incomplete_features, dead_code)]
#![feature(field_projections)]

use std::field::{UnalignedField, field_of};

pub union Foo {
    a: isize,
    b: usize,
}

type X = field_of!(Foo, a);

fn main() {
    assert_eq!(X::OFFSET, <field_of!(Foo, b)>::OFFSET);
}
