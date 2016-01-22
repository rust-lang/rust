#![feature(plugin)]
#![plugin(clippy)]

#![deny(warnings)]

#[derive(PartialEq, Hash)]
struct Foo;

impl PartialEq<u64> for Foo {
    fn eq(&self, _: &u64) -> bool { true }
}

#[derive(Hash)]
//~^ ERROR you are deriving `Hash` but have implemented `PartialEq` explicitely
struct Bar;

impl PartialEq for Bar {
    fn eq(&self, _: &Bar) -> bool { true }
}

#[derive(Hash)]
//~^ ERROR you are deriving `Hash` but have implemented `PartialEq` explicitely
struct Baz;

impl PartialEq<Baz> for Baz {
    fn eq(&self, _: &Baz) -> bool { true }
}

fn main() {}
