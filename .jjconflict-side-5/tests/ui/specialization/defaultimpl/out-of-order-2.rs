//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for the ICE in #160994, caused by an ordering issue in the new trait solver.

#![feature(specialization)]

trait Foo {
    type Out;
}

impl Foo for bool {
    type Out = ();
}

default impl<T> Foo for T {}

fn main() {}
