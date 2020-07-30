// run-pass
// Verifies that having generic parameters after constants is permitted

#![feature(const_generics)]
#![allow(incomplete_features)]

struct A<const N: usize, T=u32>(T);

fn main() {
  let _: A<3> = A(0);
}
