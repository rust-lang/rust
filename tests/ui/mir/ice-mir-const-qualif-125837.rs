// Test for ICE: mir_const_qualif: index out of bounds: the len is 0 but the index is 0
// https://github.com/rust-lang/rust/issues/125837

use std::fmt::Debug;

trait Foo<Item> {}

impl<Item, D: Debug + Clone> Foo for D {
//~^ ERROR missing generics for trait `Foo`
    fn foo<'a>(&'a self) -> impl Debug {
    //~^ ERROR method `foo` is not a member of trait `Foo`
        const { return }
//~^ ERROR return statement outside of function body
    }
}

pub fn main() {}
