#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    fn bar() -> impl std::fmt::Display;
}

impl Foo for () {
    fn bar() -> () {}
    //~^ ERROR `()` doesn't implement `std::fmt::Display`
}

fn main() {}
