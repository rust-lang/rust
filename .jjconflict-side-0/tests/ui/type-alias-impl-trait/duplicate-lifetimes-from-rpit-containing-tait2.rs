//@ check-pass
//@ edition: 2021

#![feature(type_alias_impl_trait)]

struct Foo<'a>(&'a ());

impl<'a> Foo<'a> {
    async fn new() -> () {
        type T = impl Sized;
        let _: T = ();
    }
}

fn main() {}
