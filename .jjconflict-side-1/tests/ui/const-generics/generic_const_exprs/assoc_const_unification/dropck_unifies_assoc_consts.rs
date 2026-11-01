//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}

struct Foo<T: Trait>(T)
where
    [(); T::ASSOC]:;

impl<T: Trait> Drop for Foo<T>
where
    [(); T::ASSOC]:,
{
    fn drop(&mut self) {}
}

fn main() {}
