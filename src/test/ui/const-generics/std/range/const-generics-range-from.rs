// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// `RangeFrom` should be usable within const generics:

struct S<const R: std::ops::RangeFrom<usize>>;

const C : S<{ 0 .. }> = S;

pub fn main() {}
