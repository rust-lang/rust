//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]

const fn x() {
    let t = true;
    let x = || t;
}

fn main() {}
