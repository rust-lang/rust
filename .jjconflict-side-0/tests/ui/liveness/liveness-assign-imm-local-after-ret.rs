//@ run-pass

#![allow(unreachable_code)]

#![allow(dead_code)]

fn test() {
    let _v: isize;
    _v = 1;
    return;
    _v = 2;
}

pub fn main() {
}
