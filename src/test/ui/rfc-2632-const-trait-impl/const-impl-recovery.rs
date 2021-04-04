#![feature(const_trait_impl)]
#![allow(incomplete_features)]

trait Foo {}

const impl Foo for i32 {} //~ ERROR: expected identifier, found keyword

trait Bar {}

const impl<T: Foo> Bar for T {} //~ ERROR: expected identifier, found keyword

const fn still_implements<T: Bar>() {}

const _: () = still_implements::<i32>();

fn main() {}
