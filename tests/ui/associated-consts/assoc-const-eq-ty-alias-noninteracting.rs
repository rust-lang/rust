// Regression test for issue #112560.
// Respect the fact that (associated) types and constants live in different namespaces and
// therefore equality bounds involving identically named associated items don't conflict if
// their kind (type vs. const) differs.

// FIXME(fmease): Extend this test to cover supertraits again
// once #118040 is fixed. See initial version of PR #118360.

//@ check-pass

#![feature(associated_const_equality)]

trait Trait {
    type N;

    const N: usize;
}

fn take(_: impl Trait<N = 0, N = ()>) {}

fn main() {}
