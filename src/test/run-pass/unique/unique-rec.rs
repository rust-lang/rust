// run-pass
#![feature(box_syntax)]

struct X { x: isize }

pub fn main() {
    let x: Box<_> = box X {x: 1};
    let bar = x;
    assert_eq!(bar.x, 1);
}
