// run-pass
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box vec![100];
    assert_eq!((*i)[0], 100);
}
