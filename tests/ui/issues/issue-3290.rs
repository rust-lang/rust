//@ run-pass
#![allow(dead_code)]

pub fn main() {
   let mut x: Box<_> = Box::new(3);
   x = x;
   assert_eq!(*x, 3);
}
