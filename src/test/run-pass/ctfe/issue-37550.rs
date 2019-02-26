// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]

#![feature(const_fn)]

const fn x() {
    let t = true;
    let x = || t;
}

fn main() {}
