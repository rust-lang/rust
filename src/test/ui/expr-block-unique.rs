// run-pass
#![allow(unused_braces)]

pub fn main() { let x: Box<_> = { Box::new(100) }; assert_eq!(*x, 100); }
