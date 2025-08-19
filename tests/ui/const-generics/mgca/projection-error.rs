#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

// Regression test for #140642. Test that normalizing const aliases
// containing erroneous types normalizes to a const error instead of
// a type error.


pub trait Tr<A> {
    const SIZE: usize;
}

fn mk_array(_x: T) -> [(); <T as Tr<bool>>::SIZE] {}
//~^ ERROR: cannot find type `T` in this scope
//~| ERROR: cannot find type `T` in this scope

fn main() {}
