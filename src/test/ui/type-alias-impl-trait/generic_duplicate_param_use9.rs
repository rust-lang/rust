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
    //~^ ERROR the trait bound `A: Foo` is not satisfied
    //~| ERROR `A` doesn't implement `Debug`
    //~| ERROR `B` doesn't implement `Debug`
}

fn three<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
    (t, u, 42)
    //~^ ERROR `A` doesn't implement `Debug`
    //~| ERROR `B` doesn't implement `Debug`
}
