//@ check-pass
//@ compile-flags: -Znext-solver

// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/1
// a minimization of a pattern in core.
fn next<T: Iterator<Item = U>, U>(t: &mut T) -> Option<U> {
    t.next()
}

fn foo<T: Iterator>(t: &mut T) {
    let _: Option<T::Item> = next(t);
}

fn main() {}
