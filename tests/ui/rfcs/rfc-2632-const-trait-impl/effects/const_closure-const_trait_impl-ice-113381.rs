//@ check-pass
// FIXME(effects) this shouldn't pass
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
    // FIXME(effects) ~^ ERROR: cannot call non-const fn
}
