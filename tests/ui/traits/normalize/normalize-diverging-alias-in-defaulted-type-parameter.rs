// Ensure that defaulted type parameters whose default contains diverging aliases are properly
// caught with both solvers. This is currently not the case, and this is tracked in issue
// https://github.com/rust-lang/rust/issues/156271.
// MCVE from https://github.com/rust-lang/trait-system-refactor-initiative/issues/139#issuecomment-2704576249.

//@ revisions: current next
//@ check-pass
//@ known-bug: #156271
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

trait Trait {
    type Diverges<D: Trait>;
}

impl<T> Trait for T {
    type Diverges<D: Trait> = D::Diverges<D>;
}

struct Bar<T = <u8 as Trait>::Diverges<u8>>(*mut T);

fn main() {}
