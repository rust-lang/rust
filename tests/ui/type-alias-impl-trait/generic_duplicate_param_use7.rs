//@ check-pass
#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<A: Debug, B> = impl Debug;

fn two<T: Debug + Copy, U>(t: T, u: U) -> Two<T, U> {
    (t, t)
}

fn three<T: Debug, U>(t: T, t2: T, u: U) -> Two<T, U> {
    (t, t2)
}

fn four<T: Debug, U, V>(t: T, t2: T, u: U, v: V) -> Two<T, U> {
    (t, t2)
}

fn five<X, Y: Debug>(x: X, y: Y, y2: Y) -> Two<Y, X> {
    (y, y2)
}
