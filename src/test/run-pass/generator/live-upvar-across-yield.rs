// run-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let b = |_| 3;
    let mut a = || {
        b(yield);
    };
    Pin::new(&mut a).resume();
}
