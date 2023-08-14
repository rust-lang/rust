// gate-test-impl_exhaustive_const_traits
// [unified] run-pass
// revisions: unified normal
#![feature(generic_const_exprs)]
#![cfg_attr(unified, feature(impl_exhaustive_const_traits))]
#![allow(incomplete_features)]

pub const fn bool_to_usize(b: bool) -> usize {
    b as usize
}

pub struct ConstOption<T, const S: bool> where [T; bool_to_usize(S)]:, {
    _v: [T; bool_to_usize(S)]
}

impl<T: Default> Default for ConstOption<T, true> {
    fn default() -> Self {
        Self {
        _v: [T::default()]
        }
    }
}

impl<T: Default> Default for ConstOption<T, false> {
    fn default() -> Self {
        Self {
            _v: [],
        }
    }
}

fn _test_func<const N: usize>() where ConstOption<usize, {N >= 5}>: Default,
    [usize; bool_to_usize(N >= 5)]: {

}

#[derive(Default)]
pub struct Arg<const N: usize> where
  [(); bool_to_usize(N <= 0)]:, [(); bool_to_usize(N <= 1)]:,
  [(); bool_to_usize(N <= 5)]:, [(); bool_to_usize(N > 3)]:,
  {
    _a: ConstOption<usize, { N <= 0 }>,
    //[normal]~^ ERROR: the trait
    _b: ConstOption<usize, { N <= 1 }>,
    //[normal]~^ ERROR: the trait
}

fn main() {
  let _def = Arg::<2>::default();
}
