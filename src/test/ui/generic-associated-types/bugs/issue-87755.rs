// check-fail
// known-bug

// This should pass.

#![feature(generic_associated_types)]

use std::fmt::Debug;

trait Foo {
    type Ass where Self::Ass: Debug;
}

#[derive(Debug)]
struct Bar;

impl Foo for Bar {
    type Ass = Bar;
}

fn main() {}
