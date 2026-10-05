//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(gca_min_const_items)]
#![feature(gca_const_items)]
//[current]~^ ERROR next-solver
fn main() {}
