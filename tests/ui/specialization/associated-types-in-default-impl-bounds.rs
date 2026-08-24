//@ check-pass

#![allow(incomplete_features)]
#![feature(specialization)]

// Tests that you can use a trait's associated types in the bounds of a default impl.
// Regression test for #52396.

trait Foo {
    type Baz;
    fn bar(&self, _: Self::Baz);
}

default impl<A: Foo<Baz = isize>> Foo for A {
    fn bar(&self, _: isize) { }
}

fn main() {}
