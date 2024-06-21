//@ check-pass
#![feature(const_trait_impl, effects)] //~ WARN the feature `effects` is incomplete

const fn a() {}

fn foo<F: FnOnce()>(a: F) {}

fn main() {
    let _ = a;
    foo(a);
}
