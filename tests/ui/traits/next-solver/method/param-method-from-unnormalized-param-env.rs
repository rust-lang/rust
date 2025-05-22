//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/214>.

fn execute<K, F, R>(q: F::Item) -> R
where
    F: Iterator<Item = R>,
    // Both of the below bounds should be considered for `.into()`, and then be combined
    // into a single `R: Into<?0>` bound which can be inferred to `?0 = R`.
    F::Item: Into<K>,
    R: Into<String>,
{
    q.into()
}

fn main() {}
