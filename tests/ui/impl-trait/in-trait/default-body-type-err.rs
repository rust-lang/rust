// known-bug: #108142

#![allow(incomplete_features)]
#![feature(return_position_impl_trait_in_trait)]

use std::ops::Deref;

pub trait Foo {
    fn lol(&self) -> impl Deref<Target = String> {
        &1i32
    }
}

fn main() {}
