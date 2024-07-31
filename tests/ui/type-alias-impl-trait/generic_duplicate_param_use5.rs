#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

#[defines(Two)]
fn two<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
    (t, u)
}

#[defines(Two)]
fn three<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
    (u, t)
    //~^ ERROR concrete type differs
}
