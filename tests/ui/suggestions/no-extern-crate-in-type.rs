//@ aux-build:foo.rs

extern crate foo;

type Output = Option<Foo>; //~ ERROR cannot find type `Foo`

fn main() {}
