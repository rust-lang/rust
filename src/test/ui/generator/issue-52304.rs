// check-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;

pub fn example() -> impl Generator {
    || yield &1
}

fn main() {}
