#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo {
    fn baz(&self) -> impl Debug;
}

impl Foo for u32 {
    fn baz(&self) -> u32 {
        32
    }
}

fn main() {
    let i = Box::new(42_u32) as Box<dyn Foo>;
    //~^ ERROR the trait `Foo` cannot be made into an object
    //~| ERROR the trait `Foo` cannot be made into an object
    let s = i.baz();
    //~^ ERROR the trait `Foo` cannot be made into an object
}
