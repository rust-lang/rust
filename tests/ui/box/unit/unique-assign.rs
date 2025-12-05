//@ run-pass
#![allow(unused_mut)]

pub fn main() {
    let mut i: Box<_>;
    i = Box::new(1);
    assert_eq!(*i, 1);
}
