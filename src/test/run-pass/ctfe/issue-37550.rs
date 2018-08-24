#![feature(const_fn, const_let)]

const fn x() {
    let t = true;
    let x = || t;
}

fn main() {}
