#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

#[define_opaque(Two)]
fn two<T: Debug, U>(t: T, _: U) -> Two<T, U> {
    t
}

#[define_opaque(Two)]
fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
    //~^ ERROR concrete type differs
    u
}
