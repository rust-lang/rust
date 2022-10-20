#![feature(effects)]

// check-pass

fn main() {}

const fn f() {
    g();
    let x = g;
    x();
}

const fn g() {}
