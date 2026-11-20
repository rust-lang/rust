//! Regression test for <https://github.com/rust-lang/rust/issues/17662>.
//@ run-pass
//@ aux-build:dyn-trait-lifetime-infer-metadata.rs

extern crate dyn_trait_lifetime_infer_metadata as i;

use std::marker;

struct Bar<'a> { m: marker::PhantomData<&'a ()> }

impl<'a> i::Foo<'a, usize> for Bar<'a> {
    fn foo(&self) -> usize { 5 }
}

pub fn main() {
    assert_eq!(i::foo(&Bar { m: marker::PhantomData }), 5);
}
