//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#204, see
// the sibling test for more details.

fn constrain<'a, F: FnOnce(&'a ())>(_: F) {}
fn foo<'a>(_: &'a ()) -> impl Sized + use<'a> {
    constrain(foo);
    ()
}

fn main() {}
