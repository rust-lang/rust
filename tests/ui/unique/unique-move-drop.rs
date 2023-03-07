// run-pass

#![allow(unused_variables)]

pub fn main() {
    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(200);
    let j = i;
    assert_eq!(*j, 100);
}
