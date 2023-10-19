// check-pass

#![feature(generators, generator_trait)]

use std::ops::Coroutine;

pub fn example() -> impl Coroutine {
    || yield &1
}

fn main() {}
