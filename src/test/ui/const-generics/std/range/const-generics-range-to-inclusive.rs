// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// `RangeToInclusive` should be usable within const generics:

struct S<const R: std::ops::RangeToInclusive<usize>>;

const C : S<{ ..= 999 }> = S;

pub fn main() {}
