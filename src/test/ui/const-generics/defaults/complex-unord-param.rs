// run-pass
// Checks a complicated usage of unordered params

#![feature(const_generics)]
#![allow(incomplete_features)]
#![allow(dead_code)]

struct FixedOutput<'a, const N: usize, T=u32> {
  out: &'a [T; N],
}

trait FixedOutputter {
  fn out(&self) -> FixedOutput<'_, 10>;
}

struct NestedArrays<'a, const N: usize, A: 'a, const M: usize, T:'a =u32> {
  args: &'a [&'a [T; M]; N],
  specifier: A,
}

fn main() {
  let array = [1, 2, 3];
  let nest = [&array];
  let _ = NestedArrays {
    args: &nest,
    specifier: true,
  };
}
