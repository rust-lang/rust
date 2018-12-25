// run-pass
#![feature(box_syntax)]

pub fn main() {
    let mut i: Box<_> = box 1;
    // Should be a copy
    let mut j;
    j = i.clone();
    *i = 2;
    *j = 3;
    assert_eq!(*i, 2);
    assert_eq!(*j, 3);
}
