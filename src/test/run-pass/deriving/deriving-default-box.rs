// run-pass
#![feature(box_syntax)]

use std::default::Default;

#[derive(Default)]
struct A {
    foo: Box<[bool]>,
}

pub fn main() {
    let a: A = Default::default();
    let b: Box<[_]> = Box::<[bool; 0]>::new([]);
    assert_eq!(a.foo, b);
}
