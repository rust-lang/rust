// Check that tail call checks correctly handle projections in signatures.
// (this was briefly broken while working on #156007)
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

trait Trait {
    type Assoc;
}

fn a<T: Trait>(_: T::Assoc) {}

fn b<T: Trait>(x: T::Assoc) {
    become a::<T>(x)
}

fn main() {}
