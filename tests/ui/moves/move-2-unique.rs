//@ run-pass
#![allow(dead_code)]

struct X { x: isize, y: isize, z: isize }

pub fn main() {
    let x: Box<_> = Box::new(X {x: 1, y: 2, z: 3});
    let y = x;
    assert_eq!(y.y, 2);
}
