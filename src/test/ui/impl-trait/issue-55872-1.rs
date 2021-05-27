// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S: Default> Bar for S {
    type E = impl Copy;
    //~^ ERROR the trait bound `S: Copy` is not satisfied in `(S, T)` [E0277]
    //~^^ ERROR the trait bound `T: Copy` is not satisfied in `(S, T)` [E0277]

    fn foo<T: Default>() -> Self::E {
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        //~| ERROR impl has stricter requirements than trait
        (S::default(), T::default())
    }
}

fn main() {}
