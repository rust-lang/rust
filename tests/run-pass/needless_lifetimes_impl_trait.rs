#![feature(plugin)]
#![plugin(clippy)]
#![feature(conservative_impl_trait)]
#![deny(needless_lifetime)]

trait Foo {}

struct Bar {}

struct Baz<'a> {
    bar: &'a Bar,
}

impl<'a> Foo for Baz<'a> {}

impl Bar {
    fn baz<'a>(&'a self) -> impl Foo + 'a {
        Baz { bar: self }
    }
}

fn main() {}
