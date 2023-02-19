// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018

#![feature(type_alias_impl_trait)]

pub trait Bar {
    type E: Send;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl std::marker::Send;
    fn foo<T>() -> Self::E {
        async {}
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        //[drop_tracking_mir]~^^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
    }
}

fn main() {}
