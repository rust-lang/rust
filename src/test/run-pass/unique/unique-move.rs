// run-pass
#![allow(unused_mut)]
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 100;
    let mut j;
    j = i;
    assert_eq!(*j, 100);
}
