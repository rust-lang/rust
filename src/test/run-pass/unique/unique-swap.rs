// run-pass
#![feature(box_syntax)]

use std::mem::swap;

pub fn main() {
    let mut i: Box<_> = box 100;
    let mut j: Box<_> = box 200;
    swap(&mut i, &mut j);
    assert_eq!(i, box 200);
    assert_eq!(j, box 100);
}
