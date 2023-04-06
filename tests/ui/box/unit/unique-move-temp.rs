// run-pass
#![allow(unused_mut)]

pub fn main() {
    let mut i: Box<_>;
    i = Box::new(100);
    assert_eq!(*i, 100);
}
