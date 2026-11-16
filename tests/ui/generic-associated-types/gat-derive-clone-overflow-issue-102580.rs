//@ edition: 2021
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/102580>.
// Used to overflow

trait Trait: Clone {
    type Type<T: Clone>: Clone;
}

#[derive(Clone)]
struct Struct<T: Trait>(Option<T::Type<Self>>);

fn main() {}
