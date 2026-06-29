#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Regression test for https://github.com/rust-lang/rust/issues/156701.

struct Type;

enum TypeTreeValueIter<T> {
    A(T),
}

impl<T> TypeTreeValueIter<Type> {
    //~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
    type Item = T;
}

fn main() {}
