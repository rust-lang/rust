//@ compile-flags: -Znext-solver
//@ run-pass
//@ known-bug: #130516

// `type_name` includes the unsafe binder.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::any::type_name;

fn main() {
    assert_eq!(type_name::<unsafe<'a> &'a ()>(), "&'_ ()");
    assert_eq!(type_name::<unsafe<'a> (&'a u8, u8)>(), "(&'_ u8, u8)");
}
