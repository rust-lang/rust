#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<A, B> = impl Debug;

trait Foo {
    type Bar: Debug;
    const BAR: Self::Bar;
}

fn two<T: Debug + Foo, U: Debug>(t: T, u: U) -> Two<T, U> {
    (t, u, T::BAR)
}

fn three<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
    (t, u, 42)
    //~^ ERROR concrete type differs
}
