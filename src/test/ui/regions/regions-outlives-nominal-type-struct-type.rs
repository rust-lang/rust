// Test that a nominal type (like `Foo<'a>`) outlives `'b` if its
// arguments (like `'a`) outlive `'b`.
//
// Rule OutlivesNominalType from RFC 1214.

// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![allow(dead_code)]

mod variant_struct_type {
    struct Foo<T> {
        x: T
    }
    struct Bar<'a,'b> {
        f: &'a Foo<&'b i32> //~ ERROR reference has a longer lifetime
    }
}

fn main() { }
