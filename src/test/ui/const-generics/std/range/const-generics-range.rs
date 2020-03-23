// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// `Range` should be usable within const generics:

struct S<const R: std::ops::Range<usize>>;

const C : S<{ 0 .. 1000 }> = S;

pub fn main() {}
