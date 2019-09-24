// run-pass
#![feature(box_syntax)]

pub fn main() {
    let mut i: Box<_> = box 0;
    *i = 1;
    assert_eq!(*i, 1);
}
