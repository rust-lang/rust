//! Regression test for https://github.com/rust-lang/rust/issues/2463

//@ run-pass
#![allow(dead_code)]

struct Pair {
    f: isize,
    g: isize,
}

pub fn main() {
    let x = Pair { f: 0, g: 0 };

    let _y = Pair { f: 1, g: 1, ..x };

    let _z = Pair { f: 1, ..x };
}
