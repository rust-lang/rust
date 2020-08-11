// check-pass

#![feature(const_generics)]
#![feature(const_generic_defaults)]
#![allow(incomplete_features)]


#[derive(Default)]
pub struct ConstDefault<const N: usize = 3> {
  items: [u32; N]
}

pub fn main() {
  let s = ConstDefault::default();
}
