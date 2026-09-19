//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]
#![feature(const_destruct)]

use std::marker::Destruct;

struct Wrap<T>(*const T);

const impl<T: [const] Destruct> Drop for Wrap<T> {
    fn drop(&mut self) {}
}

enum Foo {
    A,
    B(Wrap<Self>),
}

const fn drop_foo(_: Foo) {}

fn main() {}
