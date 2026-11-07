//! Reporting the unconstrained const param error used to hit
//! `assertion failed: !origin.span.is_dummy()` with the new solver.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/152415>.
//@ compile-flags: -Znext-solver

trait Super<T> {
    type Assoc;
}

trait Mirror {
    type Assoc;
}

impl<T, const N: usize> Mirror for T {
    //~^ ERROR the const parameter `N` is not constrained by the impl trait, self type, or predicates
    type Assoc = T;
}

trait Foo<A>: Super<<A as Mirror>::Assoc, Assoc = A> {}

fn main() {}
