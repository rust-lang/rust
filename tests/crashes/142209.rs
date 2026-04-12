//@ known-bug: rust-lang/rust#142209
#![feature(generic_const_exprs)]

struct Bar<const X: usize>;
const FRAC_LHS: usize = 0;

trait Foo<const N: usize> {}

impl<const N: usize = { const { 3 } }> PartialEq<dyn Foo<FRAC_LHS>> for Bar<KABOOM> {}

pub fn main() {}
