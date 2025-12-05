#![deny(clippy::elidable_lifetime_names)]
#![allow(dead_code)]

trait Foo {}

struct Bar;

struct Baz<'a> {
    bar: &'a Bar,
}

impl<'a> Foo for Baz<'a> {}
//~^ elidable_lifetime_names

impl Bar {
    fn baz<'a>(&'a self) -> impl Foo + 'a {
        //~^ elidable_lifetime_names

        Baz { bar: self }
    }
}

fn main() {}
