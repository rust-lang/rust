// check-pass

#![feature(const_generics)]
#![feature(const_generic_defaults)]
#![allow(incomplete_features)]


pub struct ConstDefault<const N: usize = 3> {}

pub fn main() {
  let s = ConstDefault::default();
}
