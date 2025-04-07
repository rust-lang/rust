#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<T, U> = impl Debug;

#[define_opaque(Two)]
fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}

#[define_opaque(Two)]
fn three<T: Debug, U: Debug>(_: T, u: U) -> Two<T, U> {
    //~^ concrete type differs
    (u, 4u32)
}
