#![feature(impl_trait_in_assoc_type)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S: Default> Bar for S {
    type E = impl Copy;

    fn foo<T: Default>() -> Self::E {
        //~^ ERROR impl has stricter requirements than trait
        //~| ERROR the trait bound `S: Copy` is not satisfied in `(S, T)` [E0277]
        //~| ERROR the trait bound `T: Copy` is not satisfied in `(S, T)` [E0277]
        //~| ERROR type parameter `T` is part of concrete type
        (S::default(), T::default())
    }
}

fn main() {}
