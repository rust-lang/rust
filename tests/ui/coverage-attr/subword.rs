#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.syntax

// Check that yes/no in `#[coverage(yes)]` and `#[coverage(no)]` must be bare
// words, not part of a more complicated substructure.

#[coverage(yes(milord))] //~ ERROR malformed `coverage` attribute input
fn yes_list() {}

#[coverage(no(milord))] //~ ERROR malformed `coverage` attribute input
fn no_list() {}

#[coverage(yes = "milord")] //~ ERROR malformed `coverage` attribute input
fn yes_key() {}

#[coverage(no = "milord")] //~ ERROR malformed `coverage` attribute input
fn no_key() {}

fn main() {}
