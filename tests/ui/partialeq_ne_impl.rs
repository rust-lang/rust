#![feature(plugin)]
#![plugin(clippy)]

#![deny(warnings)]
#![allow(dead_code)]

struct Foo;

impl PartialEq for Foo {
    fn eq(&self, _: &Foo) -> bool { true }
    fn ne(&self, _: &Foo) -> bool { false }
}

fn main() {}
