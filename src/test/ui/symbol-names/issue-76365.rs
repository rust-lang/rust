// check-pass
// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy --crate-type=lib
    //[v0]compile-flags: -Z symbol-mangling-version=v0 --crate-type=lib

#![feature(min_const_generics)]

pub struct Bar<const F: bool>;

impl Bar<true> {
    pub fn foo() {}
}

impl<const F: bool> Bar<F> {
    pub fn bar() {}
}

fn main() {}
