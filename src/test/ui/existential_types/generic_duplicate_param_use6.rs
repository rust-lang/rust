#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: Debug;

fn two<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
    (t, t)
}

fn three<T: Debug, U: Debug>(t: T, u: U) -> Two<T, U> {
//~^ concrete type differs from previous
    (u, t)
}
