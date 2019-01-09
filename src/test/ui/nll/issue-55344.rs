// compile-pass

#![feature(nll)]
#![allow(unreachable_code)]
#![deny(unused_mut)]

pub fn foo() {
    return;

    let mut v = 0;
    assert_eq!(v, 0);
    v = 1;
    assert_eq!(v, 1);
}

fn main() {}
