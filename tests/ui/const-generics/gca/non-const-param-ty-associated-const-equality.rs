//@ check-pass
//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
#![allow(dead_code)]

enum Foo {
    A,
    B,
    C,
    D(fn()),
}

trait Trait {
    const X: Foo;
}

fn foo(_: impl Trait<X = { Foo::A }>) {}

fn main() {}
