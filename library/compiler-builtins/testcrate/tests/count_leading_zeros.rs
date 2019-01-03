#![feature(compiler_builtins_lib)]

extern crate compiler_builtins;

use compiler_builtins::int::__clzsi2;

#[test]
fn __clzsi2_test() {
  let mut i: usize = core::usize::MAX;
  // Check all values above 0
  while i > 0 {
    assert_eq!(__clzsi2(i) as u32, i.leading_zeros());
    i >>= 1;
  }
  // check 0 also
  i = 0;
  assert_eq!(__clzsi2(i) as u32, i.leading_zeros());
  // double check for bit patterns that aren't just solid 1s
  i = 1;
  for _ in 0..63 {
    assert_eq!(__clzsi2(i) as u32, i.leading_zeros());
    i <<= 2;
    i += 1;
  }
}
