#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

existential type Two<T, U>: Debug;

fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}

fn three<T: Debug, U: Debug>(_: T, u: U) -> Two<T, U> {
//~^ concrete type differs from previous
    (u, 4u32)
}
