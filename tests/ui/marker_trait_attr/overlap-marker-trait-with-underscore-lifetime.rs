//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

#![feature(marker_trait_attr)]

#[marker]
trait Marker {}

impl Marker for &'_ () {} //[current]~ ERROR type annotations needed
impl Marker for &'_ () {} //[current]~ ERROR type annotations needed

fn main() {}
