//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;

mod foo {
    pub struct Struct {
        a: i32,
        pub b: i32,
    }

    pub union Union {
        a: i32,
        pub b: u32,
    }

    pub enum Enum {
        A { field: i32 },
        B(i32),
    }
}

use foo::{Enum, Struct, Union};

fn main() {
    let _: field_of!(Struct, a); //~ ERROR: field `a` of struct `Struct` is private [E0616]
    let _: field_of!(Struct, b);
    let _: field_of!(Union, a); //~ ERROR: field `a` of union `foo::Union` is private [E0616]
    let _: field_of!(Union, b);
    let _: field_of!(Enum, A.field);
    let _: field_of!(Enum, B.0);
}
