#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, UnalignedField, field_of};

mod foo {
    pub struct A {
        a: usize,
        pub b: B,
    }

    pub struct B {
        b: usize,
        pub(crate) c: C,
    }

    pub struct C {
        c: usize,
        pub(super) d: bar::D,
    }

    pub mod bar {
        pub struct D {
            d: usize,
            pub e: E,
        }

        pub struct E {
            pub(super) e: usize,
            pub(crate) f: F,
        }

        pub struct F {
            f: usize,
            pub(in super::super) g: (),
        }
    }
}

fn main() {
    use foo::A;

    let _: field_of!(A, a); //~ ERROR: field `a` of struct `A` is private [E0616]
    let _: field_of!(A, b);
    let _: field_of!(A, b.b); //~ ERROR: field `b` of struct `B` is private [E0616]
    let _: field_of!(A, b.c);
    let _: field_of!(A, b.c.c); //~ ERROR: field `c` of struct `C` is private [E0616]
    let _: field_of!(A, b.c.d);
    let _: field_of!(A, b.c.d.d); //~ ERROR: field `d` of struct `D` is private [E0616]
    let _: field_of!(A, b.c.d.e);
    let _: field_of!(A, b.c.d.e.e); //~ ERROR: field `e` of struct `E` is private [E0616]
    let _: field_of!(A, b.c.d.e.f);
    let _: field_of!(A, b.c.d.e.f.f); //~ ERROR: field `f` of struct `F` is private [E0616]
    let _: field_of!(A, b.c.d.e.f.g);
}
