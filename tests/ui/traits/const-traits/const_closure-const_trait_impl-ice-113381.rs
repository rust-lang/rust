//@ known-bug: #110395
//@ compile-flags: -Znext-solver
#![feature(const_closures, const_trait_impl)]
#![allow(incomplete_features)]

trait Foo {
    fn foo(&self);
}

impl Foo for () {
    fn foo(&self) {}
}

fn main() {
    (const || (()).foo())();
    // ^ ERROR: cannot call non-const method `<() as Foo>::foo` in constant functions
    // FIXME(const_trait_impl) this should probably say constant closures
}
