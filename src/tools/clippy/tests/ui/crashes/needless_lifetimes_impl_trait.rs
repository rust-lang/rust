#![deny(clippy::needless_lifetimes)]
#![allow(dead_code)]

trait Foo {}

struct Bar;

struct Baz<'a> {
    bar: &'a Bar,
}

impl<'a> Foo for Baz<'a> {}

impl Bar {
    fn baz<'a>(&'a self) -> impl Foo + 'a {
        //~^ ERROR: the following explicit lifetimes could be elided: 'a
        Baz { bar: self }
    }
}

fn main() {}
