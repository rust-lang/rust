//@ check-pass

// https://github.com/rust-lang/rust/pull/124840#issuecomment-2098148587

mod a {
    pub(crate) use crate::S;
}
mod b {
    pub struct S;
}
use self::a::S;
use self::b::*;

fn main() {}
