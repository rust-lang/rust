// run-pass
#![allow(unused_mut)]
#![feature(box_syntax)]

#[derive(Clone)]
struct Triple {
    x: isize,
    y: isize,
    z: isize,
}

fn test(x: bool, foo: Box<Triple>) -> isize {
    let bar = foo;
    let mut y: Box<Triple>;
    if x { y = bar; } else { y = box Triple{x: 4, y: 5, z: 6}; }
    return y.y;
}

pub fn main() {
    let x: Box<_> = box Triple{x: 1, y: 2, z: 3};
    assert_eq!(test(true, x.clone()), 2);
    assert_eq!(test(true, x.clone()), 2);
    assert_eq!(test(true, x.clone()), 2);
    assert_eq!(test(false, x), 5);
}
