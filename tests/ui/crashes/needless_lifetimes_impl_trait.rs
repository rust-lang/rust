#![deny(clippy::needless_lifetimes)]
#![allow(dead_code)]

trait Foo {}

struct Bar;

struct Baz<'a> {
    bar: &'a Bar,
}

impl<'a> Foo for Baz<'a> {}
//~^ needless_lifetimes

impl Bar {
    fn baz<'a>(&'a self) -> impl Foo + 'a {
    //~^ needless_lifetimes

        Baz { bar: self }
    }
}

fn main() {}
