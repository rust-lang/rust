// issue: #87755
// check-pass

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
