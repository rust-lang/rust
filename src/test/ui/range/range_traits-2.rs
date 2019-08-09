use std::ops::*;

#[derive(Copy, Clone)] //~ ERROR Copy
struct R(Range<usize>);

fn main() {}
