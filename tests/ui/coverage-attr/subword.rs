#![feature(coverage_attribute)]
//@ edition: 2021

// Check that yes/no in `#[coverage(yes)]` and `#[coverage(no)]` must be bare
// words, not part of a more complicated substructure.

#[coverage(yes(milord))] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn yes_list() {}

#[coverage(no(milord))] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn no_list() {}

#[coverage(yes = "milord")] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn yes_key() {}

#[coverage(no = "milord")] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn no_key() {}

fn main() {}
