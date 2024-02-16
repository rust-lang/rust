//@ run-pass
#![allow(unused_mut)]

pub fn main() {
    let i: Box<_> = Box::new(100);
    let mut j;
    j = i;
    assert_eq!(*j, 100);
}
