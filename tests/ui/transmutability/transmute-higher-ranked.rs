// Ensure we don't ICE when transmuting higher-ranked types via a
// higher-ranked transmute goal.

//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#152
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
