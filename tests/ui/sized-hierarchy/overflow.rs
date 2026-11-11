//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::marker::PhantomData;

trait ParseTokens {
    type Output;
}
impl<T: ParseTokens + ?Sized> ParseTokens for Box<T> {
    type Output = ();
}

struct Element(<Box<Box<Element>> as ParseTokens>::Output);
//~^ ERROR: overflow
//[next]~| ERROR: overflow
impl ParseTokens for Element {
//~^ ERROR: overflow
    type Output = ();
}
