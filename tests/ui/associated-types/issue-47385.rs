//@ check-pass

#![feature(associated_type_defaults)]

pub struct Foo;

pub trait Bar: From<<Self as Bar>::Input> {
    type Input = Self;
}

impl Bar for Foo {
    // Will compile with explicit type:
    // type Input = Self;
}

fn main() {}
