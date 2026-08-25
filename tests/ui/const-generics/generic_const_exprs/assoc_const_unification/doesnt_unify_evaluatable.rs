#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}

fn foo<T: Trait, U: Trait>() where [(); U::ASSOC]:, {
    bar::<{ T::ASSOC }>();
    //~^ ERROR: unconstrained generic constant
}

fn bar<const N: usize>() {}

fn main() {}
