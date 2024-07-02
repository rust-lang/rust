//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

const fn a() {}

fn foo<F: FnOnce()>(a: F) {}

fn main() {
    let _ = a;
    foo(a);
}
