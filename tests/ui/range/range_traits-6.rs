use std::ops::*;

#[derive(Copy, Clone)]
struct R(RangeInclusive<usize>); //~ ERROR Copy

fn main() {}
