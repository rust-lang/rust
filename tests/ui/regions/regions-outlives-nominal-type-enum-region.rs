// Test that a nominal type (like `Foo<'a>`) outlives `'b` if its
// arguments (like `'a`) outlive `'b`.
//
// Rule OutlivesNominalType from RFC 1214.

// check-pass

#![feature(rustc_attrs)]
#![allow(dead_code)]

mod variant_struct_region {
    struct Foo<'a> {
        x: &'a i32,
    }
    enum Bar<'a,'b> {
        V(&'a Foo<'b>)
    }
}

fn main() { }
