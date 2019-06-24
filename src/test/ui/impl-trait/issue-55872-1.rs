// ignore-tidy-linelength
#![feature(existential_type)]

pub trait Bar
{
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S: Default> Bar for S {
    existential type E: Copy;
    //~^ ERROR the trait bound `S: std::marker::Copy` is not satisfied in `(S, T)` [E0277]
    //~^^ ERROR the trait bound `T: std::marker::Copy` is not satisfied in `(S, T)` [E0277]

    fn foo<T: Default>() -> Self::E {
    //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for existential type
        (S::default(), T::default())
    }
}

fn main() {}
