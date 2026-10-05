//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] known-bug: #122529
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr

// This example failed with the old solver in `fn rematch_impl`,
// likely due to a bug with the projection cache. Regression test
// for #122529.

pub trait Archive {
    type Archived;
}

impl<'a> Archive for <&'a [u8] as Archive>::Archived {
//[next]~^ ERROR: the trait bound `&'a [u8]: Archive` is not satisfied
//[next]~| ERROR: the trait bound `&'a [u8]: Archive` is not satisfied
//[next]~| ERROR: the trait bound `&'a [u8]: Archive` is not satisfied
//[next]~| ERROR: the trait bound `&'a [u8]: Archive` is not satisfied
    type Archived = ();
    //[next]~^ ERROR: the trait bound `&'a [u8]: Archive` is not satisfied
}

fn main() {}
