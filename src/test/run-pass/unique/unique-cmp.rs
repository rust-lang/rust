// run-pass
#![allow(unused_allocation)]
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 100;
    assert_eq!(i, box 100);
    assert!(i < box 101);
    assert!(i <= box 100);
    assert!(i > box 99);
    assert!(i >= box 99);
}
