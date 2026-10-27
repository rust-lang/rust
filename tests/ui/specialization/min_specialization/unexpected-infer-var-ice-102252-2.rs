//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #102252

// Regression test for #102252, which was fixed by switching to the new solver.

#![feature(min_specialization)]

trait Trait {}

impl<T> Trait for T {}

trait Data {
    type Elem;
}

struct DatasetIter<'a, R: Data> {
    data: &'a R::Elem,
}

pub struct ArrayBase {}

impl<'a> Trait for DatasetIter<'a, ArrayBase> {}
//[next]~^ ERROR: specialization impl does not specialize any associated items

fn main() {}
