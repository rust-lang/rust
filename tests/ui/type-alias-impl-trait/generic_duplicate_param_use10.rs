//@ check-pass
#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<T: Debug, U> = impl Debug;

#[define_opaque(Two)]
fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}
