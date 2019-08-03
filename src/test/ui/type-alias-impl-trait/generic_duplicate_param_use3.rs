#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

fn one<T: Debug>(t: T) -> Two<T, T> {
//~^ defining opaque type use restricts opaque type
    t
}

fn two<T: Debug, U>(t: T, _: U) -> Two<T, U> {
    t
}

fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
//~^ concrete type's generic parameters differ from previous defining use
    u
}
