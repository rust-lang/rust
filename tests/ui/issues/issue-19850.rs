//@ check-pass
#![allow(unused_variables)]
// Test that `<Type as Trait>::Output` and `Self::Output` are accepted as type annotations in let
// bindings


trait Int {
    fn one() -> Self;
    fn leading_zeros(self) -> usize;
}

trait Foo {
    type T : Int;

    fn test(&self) {
        let r: <Self as Foo>::T = Int::one();
        let r: Self::T = Int::one();
    }
}

fn main() {}
