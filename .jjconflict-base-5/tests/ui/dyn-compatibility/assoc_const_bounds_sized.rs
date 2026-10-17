#![feature(generic_const_items)]
#![allow(incomplete_features, dead_code)]

//@ check-pass

trait Foo {
    const BAR: bool
    where
        Self: Sized;
}

fn foo(_: &dyn Foo) {}

fn main() {}
