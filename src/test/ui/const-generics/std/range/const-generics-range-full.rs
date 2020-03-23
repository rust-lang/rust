// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

// `RangeFull` should be usable within const generics:

struct S<const R: std::ops::RangeFull>;

const C : S<{ .. }> = S;

pub fn main() {}
