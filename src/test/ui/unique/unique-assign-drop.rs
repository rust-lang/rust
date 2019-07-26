// run-pass
#![allow(unused_assignments)]

#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 1;
    let mut j: Box<_> = box 2;
    // Should drop the previous value of j
    j = i;
    assert_eq!(*j, 1);
}
