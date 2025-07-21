//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass
//@[next] check-pass
//@[next] compile-flags: -Znext-solver

// FIXME(sized_hierarchy): this is expected to fail in the old solver when there
// isn't a temporary revert of the `sized_hierarchy` feature

use std::marker::PhantomData;

trait ParseTokens {
    type Output;
}
impl<T: ParseTokens + ?Sized> ParseTokens for Box<T> {
    type Output = ();
}

struct Element(<Box<Box<Element>> as ParseTokens>::Output);
impl ParseTokens for Element {
    type Output = ();
}
