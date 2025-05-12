//@ pp-exact

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

struct Foo;

impl Foo {
    fn baz(&pin mut self) {}

    fn baz_const(&pin const self) {}

    fn baz_lt<'a>(&'a pin mut self) {}

    fn baz_const_lt(&'_ pin const self) {}
}

fn foo(_: &pin mut Foo) {}
fn foo_lt<'a>(_: &'a pin mut Foo) {}

fn foo_const(_: &pin const Foo) {}
fn foo_const_lt(_: &'_ pin const Foo) {}

fn main() {}
