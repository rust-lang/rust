#![crate_type = "lib"]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

const fn complex_maths<T>(n : usize) -> usize {
  2 * n + 1
}

struct Example<T, const N: usize> {
  a: [f32; N],
  b: [f32; complex_maths::<T>(N)],
  //~^ ERROR unconstrained
  c: T,
}
