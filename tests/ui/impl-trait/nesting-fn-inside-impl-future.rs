//! Regression test for <https://github.com/rust-lang/rust/issues/165091>

//@ check-pass
//@ edition: 2024

#![deny(refining_impl_trait)]

trait A { fn f() -> impl B<fn(&())>; }
impl A for () { fn f() -> impl B<fn(&())> { 0 } }

trait B<T: ?Sized> {}
impl<T: ?Sized, U: ?Sized> B<T> for U {}

fn main() {}
