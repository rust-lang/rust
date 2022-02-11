#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;
//~^ ERROR `T` doesn't implement `Debug`

fn one<T: Debug>(t: T) -> Two<T, T> {
    //~^ ERROR non-defining opaque type use in defining scope
    t
}

fn two<T: Debug, U>(t: T, _: U) -> Two<T, U> {
    t
}

fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
    //~^ ERROR concrete type differs from previous defining opaque type use
    u
}
