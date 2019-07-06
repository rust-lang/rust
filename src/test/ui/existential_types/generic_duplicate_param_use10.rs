// build-pass (FIXME(62277): could be check-pass?)
#![feature(existential_type)]

use std::fmt::Debug;

fn main() {}

existential type Two<T, U>: Debug;

fn two<T: Debug, U: Debug>(t: T, _: U) -> Two<T, U> {
    (t, 4u32)
}
