//@ run-pass

#![allow(unused_must_use)]

pub fn main() {
    let x: isize = 8;
    let y = 9;
    x + y;

    let q: isize = -8;
    let r = -9;
    q + r;
}
