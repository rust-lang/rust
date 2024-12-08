#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<T, U> = impl Debug;

fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}

fn three<T: Debug, U: Debug>(_: T, u: U) -> Two<T, U> {
    (u, 4u32)
    //~^ concrete type differs
}
