//@ check-pass
//! Tests that HRTB impl selection covers type parameters not directly related
//! to the trait.
//! Test for <https://github.com/rust-lang/rust/issues/30867>

#![crate_type = "lib"]

trait Unary<T> {}
impl<T, U, F: Fn(T) -> U> Unary<T> for F {}
fn unary<F: for<'a> Unary<&'a T>, T>() {}

pub fn test<F: for<'a> Fn(&'a i32) -> &'a i32>() {
    unary::<F, i32>()
}
