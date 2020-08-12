// run-pass
// Checks some basic test cases for defaults.
#![feature(const_generics)]
#![allow(incomplete_features)]
#![allow(dead_code)]

struct FixedOutput<'a, const N: usize, T=u32> {
  out: &'a [T; N],
}

trait FixedOutputter {
  fn out(&self) -> FixedOutput<'_, 10>;
}

fn main() {}
