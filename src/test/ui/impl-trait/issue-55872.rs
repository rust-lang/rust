// ignore-tidy-linelength
#![feature(existential_type)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    existential type E: Copy;

    fn foo<T>() -> Self::E {
    //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for existential type
        || ()
    }
}

fn main() {}
