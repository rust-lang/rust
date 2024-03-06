// Regression test for #51008 -- the anonymous lifetime in `&i32` was
// being incorrectly considered part of the "elided lifetimes" from
// the impl.
//
//@ check-pass

trait A {

}

impl<F> A for F where F: FnOnce(&i32) {}

fn main() {}
