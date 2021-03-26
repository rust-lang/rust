// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

use std::fmt::Debug;

fn main() {}

type Two<T, U> = impl Debug;
//~^ ERROR `T` doesn't implement `Debug`

fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}

fn three<T: Debug, U: Debug>(_: T, u: U) -> Two<T, U> {
    //~^ concrete type differs from previous
    (u, 4u32)
}
