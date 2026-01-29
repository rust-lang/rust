use std::ops::*;

#[derive(Copy, Clone)]
struct R(Range<usize>); //~ ERROR Copy

fn main() {}
