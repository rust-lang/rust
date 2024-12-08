//@ run-pass
#![allow(non_camel_case_types)]

trait double {
    fn double(self: Box<Self>) -> usize;
}

impl double for Box<usize> {
    fn double(self: Box<Box<usize>>) -> usize { **self * 2 }
}

pub fn main() {
    let x: Box<Box<Box<Box<Box<_>>>>> = Box::new(Box::new(Box::new(Box::new(Box::new(3)))));
    assert_eq!(x.double(), 6);
}
