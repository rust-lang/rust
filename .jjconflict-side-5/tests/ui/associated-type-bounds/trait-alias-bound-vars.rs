// Check that we're successfully collecting bound vars behind trait aliases.
// Regression test for <https://github.com/rust-lang/rust/issues/152244>.
//@ check-pass
//@ needs-rustc-debug-assertions
#![feature(trait_alias)]

trait A<'a> { type X; }
trait B: for<'a> A<'a> {}
trait C = B;

fn f<T>() where T: C<X: Copy> {}
fn g<T>() where T: C<X: for<'r> A<'r>> {}

fn main() {}
