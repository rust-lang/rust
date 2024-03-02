#![feature(impl_trait_in_assoc_type)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S: Default> Bar for S {
    type E = impl Copy;

    fn foo<T: Default>() -> Self::E {
        //~^ ERROR impl has stricter requirements than trait
        //~| ERROR the trait `Copy` is not implemented for `S` in `(S, T)`
        //~| ERROR the trait `Copy` is not implemented for `T` in `(S, T)`
        (S::default(), T::default())
    }
}

fn main() {}
