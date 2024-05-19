use std::ops::*;

#[derive(Copy, Clone)] //~ ERROR Copy
struct R(RangeInclusive<usize>);

fn main() {}
