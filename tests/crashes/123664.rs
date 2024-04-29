//@ known-bug: #123664
#![feature(generic_const_exprs, effects)]
const fn with_positive<F: ~const Fn()>() {}
pub fn main() {}
