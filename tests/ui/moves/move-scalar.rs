//@ run-pass
#![allow(unused_mut)]

pub fn main() {

    let y: isize = 42;
    let mut x: isize;
    x = y;
    assert_eq!(x, 42);
}
