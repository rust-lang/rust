use std::ops::*;

#[derive(Copy, Clone)] //~ ERROR Copy
struct R(RangeFrom<usize>);

fn main() {}
