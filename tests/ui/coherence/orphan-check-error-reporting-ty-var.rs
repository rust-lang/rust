// Regression test for #132826.

// Make sure we don't try to resolve the variable `K1` in the generics of the impl
// (which only has `K2`).

pub trait MyTrait {
    type Item;
}

impl<K1> MyTrait for Vec<K1> {
    type Item = Vec<K1>;
}

impl<K2> From<Vec<K2>> for <Vec<K2> as MyTrait>::Item {}
//~^ ERROR only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
