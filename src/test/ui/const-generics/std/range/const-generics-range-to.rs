// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// `RangeTo` should be usable within const generics:

struct S<const R: std::ops::RangeTo<usize>>;

const C : S<{ .. 1000 }> = S;

pub fn main() {}
