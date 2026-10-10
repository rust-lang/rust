//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

const fn a() {}

fn foo<F: FnOnce()>(a: F) {}

fn main() {
    let _ = a;
    foo(a);
}
