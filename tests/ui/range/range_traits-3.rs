use std::ops::range;

#[derive(Copy, Clone)]
struct R(range::RangeFrom<usize>);

#[derive(Copy, Clone)] //~ ERROR Copy
struct S(range::legacy::RangeFrom<usize>);

fn main() {}
