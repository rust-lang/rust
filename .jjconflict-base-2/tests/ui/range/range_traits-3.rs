use std::ops::*;

#[derive(Copy, Clone)]
struct R(RangeFrom<usize>); //~ ERROR Copy

fn main() {}
