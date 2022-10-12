#![feature(box_syntax)]

pub fn main() {
    let x = box 10;
    let y = x;
    assert_eq!(*y, 10);
}
