// [unified] run-pass
// revisions: unified normal
#![feature(generic_const_exprs)]
#![cfg_attr(unified, feature(impl_exhaustive_const_traits))]
#![allow(incomplete_features)]

pub const fn bool_to_usize(b: bool) -> usize {
    b as usize
}

pub struct ConstTwo<const A: bool, const B: bool>;

impl Default for ConstTwo<true, true> {
  fn default() -> Self { Self }
}
impl Default for ConstTwo<true, false> {
  fn default() -> Self { Self }
}
impl Default for ConstTwo<false, true> {
  fn default() -> Self { Self }
}
impl Default for ConstTwo<false, false> {
  fn default() -> Self { Self }
}

#[derive(Default)]
pub struct Arg<const N: usize> where
  [(); bool_to_usize(N <= 5)]:, [(); bool_to_usize(N > 3)]:,
  {
    _c: ConstTwo<{ N <= 5 }, {N > 3}>,
    //[normal]~^ ERROR: the trait
}

fn main() {
  let _def = Arg::<2>::default();
}
