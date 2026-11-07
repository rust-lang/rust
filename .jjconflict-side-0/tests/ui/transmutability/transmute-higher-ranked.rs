// Ensure we don't ICE when transmuting higher-ranked types via a
// higher-ranked transmute goal.

//@ check-pass

#![feature(transmutability)]

use std::mem::TransmuteFrom;

pub fn transmute()
where
    for<'a> &'a &'a i32: TransmuteFrom<&'a &'a u32>,
{
}

fn main() {
    transmute();
}
