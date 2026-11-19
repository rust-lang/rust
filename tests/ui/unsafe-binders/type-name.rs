//@ compile-flags: -Znext-solver
//@ run-pass

// `type_name` includes the unsafe binder.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::any::type_name;

fn main() {
    assert_eq!(type_name::<unsafe<'a> &'a ()>(), "unsafe<> &'_ ()");
    assert_eq!(type_name::<unsafe<'a> (&'a u8, u8)>(), "unsafe<> (&'_ u8, u8)");
}
