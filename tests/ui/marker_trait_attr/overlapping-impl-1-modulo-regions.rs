//@ known-bug: #109481
//
// While the `T: Copy` is always applicable when checking
// that the impl `impl<T: Copy> F for T {}` is well formed,
// the old trait solver can only approximate this by checking
// that there are no inference variables in the obligation and
// no region constraints in the evaluation result.
//
// Because of this we end up with ambiguity here.
#![feature(marker_trait_attr)]

#[marker]
pub trait F {}
impl<T: Copy> F for T {}
impl<T: 'static> F for T {}

fn main() {}
