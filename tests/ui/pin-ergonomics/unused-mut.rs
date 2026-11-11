//@ check-pass
//@ compile-flags: --force-warn unused-mut

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// `&pin mut self` should not trigger `unused_mut`: the `mut` qualifies the pin
// reference (like `&mut self`), not the binding. See https://github.com/rust-lang/rust/issues/142077.

struct Foo;

impl Foo {
    fn baz(&pin mut self) {}
    fn baz_const(&pin const self) {}
    fn baz_lt<'a>(&'a pin mut self) {}
    fn baz_const_lt(&'_ pin const self) {}
}

fn foo(_: &pin mut Foo) {}

fn main() {}
