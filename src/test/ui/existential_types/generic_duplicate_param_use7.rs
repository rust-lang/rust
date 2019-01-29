// compile-pass
#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

existential type Two<A, B>: Debug;

fn two<T: Debug + Copy, U>(t: T, u: U) -> Two<T, U> {
    (t, t)
}

fn three<T: Debug, U>(t: T, t2: T, u: U) -> Two<T, U> {
    (t, t2)
}
