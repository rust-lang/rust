use std::ops::range;

#[derive(Copy, Clone)]
struct R(range::Range<usize>);

#[derive(Copy, Clone)] //~ ERROR Copy
struct S(range::legacy::Range<usize>);

fn main() {}
