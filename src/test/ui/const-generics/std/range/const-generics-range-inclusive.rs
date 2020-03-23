// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// Regression test for #70155

// `RangeInclusive` should be usable within const generics:

struct S<const R: std::ops::RangeInclusive<usize>>;

const C : S<{ 0 ..= 999 }> = S;

pub fn main() {}
