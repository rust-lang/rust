//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Tests that rebasing from the concrete impl to the default impl also processes the
// `[u32; 0]: IntoIterator<Item = ?U>` predicate to constrain the `?U` impl arg.

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Spec {
    type Assoc;
}

default impl<T, U> Spec for T where T: IntoIterator<Item = U> {
    type Assoc = U;
}

impl<T> Spec for [T; 0] {}

fn main() {
    let x: <[u32; 0] as Spec>::Assoc = 1;
}
