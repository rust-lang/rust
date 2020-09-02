// ignore-tidy-linelength
#![feature(type_alias_impl_trait)]

pub trait Bar
{
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S: Default> Bar for S {
    type E = impl Copy;
    //~^ ERROR the trait bound `S: Copy` is not satisfied in `(S, T)` [E0277]
    //~^^ ERROR the trait bound `T: Copy` is not satisfied in `(S, T)` [E0277]

    fn foo<T: Default>() -> Self::E {
    //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        (S::default(), T::default())
    }
}

fn main() {}
