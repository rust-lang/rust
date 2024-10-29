//@ compile-flags: -Znext-solver
#![feature(const_closures, const_trait_impl, effects)]
#![allow(incomplete_features)]

trait Foo {
    fn foo(&self);
}

impl Foo for () {
    fn foo(&self) {}
}

fn main() {
    (const || { (()).foo() })();
    //~^ ERROR: cannot call non-const fn `<() as Foo>::foo` in constant functions
    // FIXME(effects) this should probably say constant closures
}
