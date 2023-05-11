// Test that a nominal type (like `Foo<'a>`) outlives `'b` if its
// arguments (like `'a`) outlive `'b`.
//
// Rule OutlivesNominalType from RFC 1214.

// check-pass

#![feature(rustc_attrs)]
#![allow(dead_code)]

mod rev_variant_struct_region {
    struct Foo<'a> {
        x: fn(&'a i32),
    }
    struct Bar<'a,'b> {
        f: &'a Foo<'b>
    }
}

fn main() { }
