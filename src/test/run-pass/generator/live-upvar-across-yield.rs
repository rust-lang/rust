// run-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn main() {
    let b = |_| 3;
    let mut a = || {
        b(yield);
    };
    unsafe { a.resume() };
}
