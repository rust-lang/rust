//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

mod foo {
    pub struct A {
        a: i32,
        pub b: i32,
    }
}

fn main() {
    use foo::A;

    let _: field_of!(A, a); //~ ERROR: field `a` of struct `A` is private [E0616]
    let _: field_of!(A, b);
}
