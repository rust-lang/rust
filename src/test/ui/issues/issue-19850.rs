// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused_variables)]
// Test that `<Type as Trait>::Output` and `Self::Output` are accepted as type annotations in let
// bindings

// pretty-expanded FIXME #23616

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
