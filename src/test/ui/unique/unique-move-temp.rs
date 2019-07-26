// run-pass
#![allow(unused_mut)]
#![feature(box_syntax)]

pub fn main() {
    let mut i: Box<_>;
    i = box 100;
    assert_eq!(*i, 100);
}
