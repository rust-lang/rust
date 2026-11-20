//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/214>.
// See comment below.

trait A {
    fn hello(&self) {}
}

trait B {
    fn hello(&self) {}
}

impl<T> A for T {}
impl<T> B for T {}

fn test<F, R>(q: F::Item)
where
    F: Iterator<Item = R>,
    // We want to prefer `A` for `R.hello()`
    F::Item: A,
{
    q.hello();
}

fn main() {}
