#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

#[defines(Two)]
fn two<T: Debug, U>(t: T, _: U) -> Two<T, U> {
    t
}

#[defines(Two)]
fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
    u
    //~^ ERROR concrete type differs
}
