// run-pass

#![allow(unused_variables)]
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 100;
    let j: Box<_> = box 200;
    let j = i;
    assert_eq!(*j, 100);
}
