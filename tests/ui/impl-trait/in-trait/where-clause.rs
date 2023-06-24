// check-pass
// edition: 2021

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo<Item> {
    fn foo<'a>(&'a self) -> impl Debug
    where
        Item: 'a;
}

impl<Item, D: Debug + Clone> Foo<Item> for D {
    fn foo<'a>(&'a self) -> impl Debug
    where
        Item: 'a,
    {
        self.clone()
    }
}

fn main() {}
