// [full] run-pass
// revisions: min full
// Checks some basic test cases for defaults.
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![allow(dead_code)]

struct FixedOutput<'a, const N: usize, T=u32> {
  //[min]~^ ERROR type parameters must be declared prior to const parameters
  out: &'a [T; N],
}

trait FixedOutputter {
  fn out(&self) -> FixedOutput<'_, 10>;
}

fn main() {}
