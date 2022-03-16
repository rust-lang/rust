// Regression test for #92111.
//
// The issue was that we normalize trait bounds before caching
// results of selection. Checking that `impl Tr for S` requires
// checking `S: !Drop` because it cannot overlap with the blanket
// impl. Then we save the (unsatisfied) result from checking `S: Drop`.
// Then the call to `a` checks whether `S: ~const Drop` but we normalize
// it to `S: Drop` which the cache claims to be unsatisfied.
//
// check-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

pub trait Tr {}

#[allow(drop_bounds)]
impl<T: Drop> Tr for T {}

#[derive(Debug)]
pub struct S(i32);

impl Tr for S {}

const fn a<T: ~const Drop>(t: T) {}

fn main() {
    a(S(0));
}
