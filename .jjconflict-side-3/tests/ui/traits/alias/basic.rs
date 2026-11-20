//@ run-pass

#![feature(trait_alias)]

pub trait Foo {}
pub trait FooAlias = Foo;

fn main() {}
