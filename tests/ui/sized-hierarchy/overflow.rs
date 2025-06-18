//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] check-pass
//@[next] compile-flags: -Znext-solver

use std::marker::PhantomData;

trait ParseTokens {
    type Output;
}
impl<T: ParseTokens + ?Sized> ParseTokens for Box<T> {
    type Output = ();
}

struct Element(<Box<Box<Element>> as ParseTokens>::Output);
//[current]~^ ERROR overflow evaluating
impl ParseTokens for Element {
//[current]~^ ERROR overflow evaluating
    type Output = ();
}
