//@ known-bug: #136678
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct B<const A: usize>;

struct Test<const A: usize>;

impl<const A: usize> Test<A> {
    type B = B<{ A }>;

    fn test(a: Self::B) -> Self::B {
        a
    }
}

pub fn main() {}
