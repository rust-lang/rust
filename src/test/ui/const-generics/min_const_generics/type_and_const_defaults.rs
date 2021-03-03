// check-pass
#![crate_type = "lib"]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

struct Both<T=u32, const N: usize=3> {
  arr: [T; N]
}

trait BothTrait<T=u32, const N: usize=3> {}

enum BothEnum<T=u32, const N: usize=3> {
  Dummy([T; N])
}
