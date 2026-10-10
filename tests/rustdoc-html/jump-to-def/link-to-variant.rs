// This test ensures that even if enum variants are imported (and thus present in the
// `Cache::external_paths` rustdoc data), we still link correctly to them.

#![no_std]
#![crate_name = "foo"]

//@ compile-flags: -Zunstable-options --generate-link-to-definition

//@ has 'src/foo/link-to-variant.rs.html'
//@ has - '//a[@href="{{channel}}/core/cmp/type.Ordering.html#variant.Equal"]' 'Equal'
//@ has - '//a[@href="{{channel}}/core/cmp/type.Ordering.html"]' 'Ordering'
//@ has - '//a[@href="{{channel}}/core/cmp/type.Ordering.html"]' 'self'

use core::cmp::Ordering::{self, Equal};

pub fn foo(o: Ordering) -> bool {
    match o {
        Ordering::Equal => true,
        _ => false,
    }
}
