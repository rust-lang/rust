//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #102252

// Regression test for #102252, which was fixed by switching to the new solver.

#![feature(min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
pub trait Trait {}

struct Struct
//[next]~^ ERROR: overflow evaluating the requirement `<Struct as Iterator>::Item == _`
where
    Self: Iterator<Item = <Self as Iterator>::Item>, {}

impl Trait for Struct {}
//[next]~^ ERROR: the type `Struct` is not well-formed

fn main() {}
