//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

// Ensure we do not ICE if a non-well-formed return type manages to slip past HIR typeck.
// See <https://github.com/rust-lang/rust/pull/152816> for details.

pub struct Foo<T>(T)
where
    T: Iterator,
    <T as Iterator>::Item: Default;

fn foo<T>() -> Foo<T> {
    //~^ ERROR: `T` is not an iterator
    loop {}
}

fn main() {}
