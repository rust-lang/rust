// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Tests that impls are allowed to have looser, more permissive bounds
// than the traits require.


trait A {
  fn b<C:Sync,D>(&self, x: C) -> C;
}

struct E {
 f: isize
}

impl A for E {
  fn b<F,G>(&self, _x: F) -> F { panic!() }
}

pub fn main() {}
