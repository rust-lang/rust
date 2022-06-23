// check-pass
use std::fmt::Debug;

fn a<'a>() -> impl Fn(&'a u8) -> (impl Debug + '_) {
    |x| x
}

fn _b<'a>() -> impl Fn(&'a u8) -> (impl Debug + 'a) {
    a()
}

fn main() {}
