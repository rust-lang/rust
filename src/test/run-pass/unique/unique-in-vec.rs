// run-pass
#![feature(box_syntax)]

pub fn main() {
    let vect : Vec<Box<_>> = vec![box 100];
    assert_eq!(vect[0], box 100);
}
