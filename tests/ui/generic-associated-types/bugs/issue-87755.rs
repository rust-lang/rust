//@ check-fail
//@ known-bug: #87755

// This should pass.

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
