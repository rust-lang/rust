// Regression test for #51008 -- the anonymous lifetime in `&i32` was
// being incorrectly considered part of the "elided lifetimes" from
// the impl.
//
// run-pass

#![feature(rust_2018_preview)]

trait A {

}

impl<F> A for F where F: PartialEq<fn(&i32)> { }

fn main() {}
