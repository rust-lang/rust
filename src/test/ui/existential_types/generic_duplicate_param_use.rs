#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: Debug;
//~^ could not find defining uses

fn one<T: Debug>(t: T) -> Two<T, T> {
//~^ ERROR defining existential type use restricts existential type
    t
}
