// run-pass
#![allow(unused_braces)]
#![feature(box_syntax)]

pub fn main() { let x: Box<_> = { box 100 }; assert_eq!(*x, 100); }
