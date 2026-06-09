#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait _Contains<T> {
    const does_contain: bool;
}

trait Contains<T, const Satisfied: bool> {}

trait Delegates<T> {}

impl<T, U> Delegates<U> for T where T: Contains<U, true> {}

const fn contains<A, B>() -> bool
where
    A: _Contains<B>,
{
    A::does_contain
}

impl<T, U> Contains<T, { contains::<T, U>() }> for U where T: _Contains<U> {}

fn writes_to_path<C>(cap: &C) {
    writes_to_specific_path(&cap);
    //~^ ERROR: the trait bound `(): _Contains<&C>` is not satisfied [E0277]
    //~| ERROR: unconstrained generic constant
    //~| ERROR: mismatched types [E0308]
}

fn writes_to_specific_path<C: Delegates<()>>(cap: &C) {}

fn main() {}
