//@ edition: 2021
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/126296>.
// Used to overflow

trait Unimplemented {}

struct Special<U>(U);
struct Wrapper<T>(T);

trait A {}
impl<T> A for Special<Wrapper<T>> where Special<T>: A {}
impl<T: Unimplemented> A for Special<T> {}

fn main() {}
