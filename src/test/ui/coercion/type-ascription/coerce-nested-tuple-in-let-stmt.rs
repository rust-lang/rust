// run-pass

#![feature(type_ascription)]

use std::any::type_name;
use std::assert_eq;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

fn main() {
  let arr = [5,6,7];
  let tup = (5, (3, (12, &arr : &[u32])), &arr : &[u32]);
  assert_eq!(type_of(tup), "(i32, (i32, (i32, &[u32])), &[u32])");
}
