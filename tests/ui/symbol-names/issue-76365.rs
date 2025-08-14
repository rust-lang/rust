//@ check-pass
//@ revisions: legacy v0
//@[legacy]compile-flags: -Z unstable-options -C symbol-mangling-version=legacy --crate-type=lib
//@[v0]compile-flags: -C symbol-mangling-version=v0 --crate-type=lib


pub struct Bar<const F: bool>;

impl Bar<true> {
    pub fn foo() {}
}

impl<const F: bool> Bar<F> {
    pub fn bar() {}
}

fn main() {}
