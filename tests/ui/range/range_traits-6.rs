use std::ops::range;

#[derive(Copy, Clone)]
struct R(range::RangeInclusive<usize>);

#[derive(Copy, Clone)] //~ ERROR Copy
struct S(range::legacy::RangeInclusive<usize>);

fn main() {}
