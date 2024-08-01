//@ check-pass
//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicitly enabled)

// Regression test for an ICE when trying to bootstrap rustc
// with #125343. An ambiguous goal returned a `TypeOutlives`
// constraint referencing an inference variable. This inference
// variable was created inside of the goal, causing it to be
// unconstrained in the caller. This then caused an ICE in MIR
// borrowck.

struct Foo<T>(T);
trait Extend<T> {
    fn extend<I: IntoIterator<Item = T>>(iter: I);
}

impl<T> Extend<T> for Foo<T> {
    fn extend<I: IntoIterator<Item = T>>(_: I) {
        todo!()
    }
}

impl<'a, T: 'a + Copy> Extend<&'a T> for Foo<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(iter: I) {
        <Self as Extend<T>>::extend(iter.into_iter().copied())
    }
}

fn main() {}
