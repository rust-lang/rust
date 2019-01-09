#![feature(box_syntax)]

pub fn main() { let x: Box<_> = { box 100 }; assert_eq!(*x, 100); }
