#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

#[define_opaque(Two)]
fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
    u
    //~^ ERROR `U` doesn't implement `Debug`
}
