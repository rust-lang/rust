//@ known-bug: #119299
#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq)]
struct ConstStrU(*const u8, usize);

impl ConstParamTy for &'static ConstStrU {}

impl ConstStrU {
  const fn from_bytes(bytes: &'static [u8]) -> Self {
    Self(bytes.as_ptr(), bytes.len())
  }
}

const fn chars_s<const S: &'static ConstStrU>() -> [char; 3] {
  ['a','b','c']
}

fn main() {
  const A: &'static ConstStrU = &ConstStrU::from_bytes(b"abc");
  chars_s::<A>();
}
