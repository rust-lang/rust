#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type Two<T, U> = impl Debug;

fn one<T: Debug>(t: T) -> Two<T, T> {
//~^ ERROR defining existential type use restricts existential type
    t
}

fn three<T, U: Debug>(_: T, u: U) -> Two<T, U> {
    u
}
