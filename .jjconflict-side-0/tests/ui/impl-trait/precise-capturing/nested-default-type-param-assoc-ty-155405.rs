// Regression test for <https://github.com/rust-lang/rust/issues/155405>.
// A nested generic whose default type parameter referenced an associated type used to
// panic in the `impl_trait_overcaptures` lint.
//@ check-pass

#![allow(dead_code)]

trait T {
    type A;
    fn n() -> M<(), M<(), ()>> {
        loop {}
    }
}

struct M<O, V, F = fn(&<V as T>::A) -> &O>(F, V, O);

impl<O, V, F> T for M<O, V, F> {
    type A = O;
}

impl T for () {
    type A = ();
}

fn main() {}
