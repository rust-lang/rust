//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//
// Regression test for #109481 and #84917.
//
// Overlapping marker trait impls where one has a `T: 'static` bound
// should be accepted.  A bug previously made marker trait winnowing
// order-dependent, producing a spurious E0310 here.
#![feature(marker_trait_attr)]

#[marker]
pub trait F {}
impl<T: Copy> F for T {}
impl<T: 'static> F for T {}

fn main() {}
