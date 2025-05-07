//@ build-pass
#![allow(dead_code)]
// defining static with struct that contains enum
// with &'static str variant used to cause ICE


pub enum Foo {
    Bar,
    Baz(&'static str),
}

pub static TEST: Test = Test {
    foo: Foo::Bar,
    c: 'a'
};

pub struct Test {
    foo: Foo,
    c: char,
}

fn main() {}
