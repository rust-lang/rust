//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}

fn foo<T: Trait, U: Trait>() where [(); T::ASSOC]:, {
    bar::<{ T::ASSOC }>();
}

fn bar<const N: usize>() -> [(); N] {
    [(); N]
}

fn main() {}
