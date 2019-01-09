// run-pass
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 1;
    let j = i;
    assert_eq!(*j, 1);
}
