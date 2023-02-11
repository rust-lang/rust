// Test case where an associated type is referenced from within the
// supertrait definition, and the impl makes the wrong
// associations. Issue #20220.

use std::vec::IntoIter;

pub trait Foo: Iterator<Item = <Self as Foo>::Key> {
    type Key;
}

impl Foo for IntoIter<i32> {
    type Key = u32;
    //~^ ERROR expected `IntoIter<i32>` to be an iterator that yields `u32`, but it yields `i32`
}

fn main() {}
