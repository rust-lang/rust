#![feature(type_alias_impl_trait)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl Copy;

    fn foo<T>() -> Self::E {
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        || ()
    }
}

fn main() {}
