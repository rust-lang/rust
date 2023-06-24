#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Display;

trait Foo {
    fn bar(&self) -> impl Display;
}

impl Foo for () {
    fn bar(&self) -> impl Display {
        "Hello, world"
    }
}

fn main() {
    let x: &str = ().bar();
    //~^ ERROR mismatched types
}
