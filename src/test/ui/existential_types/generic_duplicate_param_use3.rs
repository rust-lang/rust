#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: Debug;

fn one<T: Debug>(t: T) -> Two<T, T> {
//~^ defining existential type use restricts existential type
    t
}

fn two<T: Debug, U>(t: T, _: U) -> Two<T, U> {
    t
}

fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
//~^ concrete type's generic parameters differ from previous defining use
    u
}
