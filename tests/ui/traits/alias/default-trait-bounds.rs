// Check that trait alias bounds also imply default trait bounds to ensure that
// default and user-written bounds behave the same wrt. trait solving.
//
// Previously, we would only imply default trait bounds on
// *`Self`* type params, not on regular type params however.
// issue: <https://github.com/rust-lang/rust/issues/152687>

//@ check-pass

#![feature(trait_alias)]
#![feature(sized_hierarchy)] // only used for test cases (B) (C).

trait A0<T: Sized> =; // has a user-written `T: Sized` bound
fn f0<T: ?Sized>() where (): A0<T> {}

trait A1<T> =; // has a default `T: Sized` bound
fn f1<T: ?Sized>() where (): A1<T> {}

//

trait B0<T: std::marker::MetaSized> =;
fn g0<T: std::marker::PointeeSized>() where (): B0<T> {}

trait B1<T: ?Sized> =; // has a default `T: MetaSized` bound
fn g1<T: std::marker::PointeeSized>() where (): B1<T> {}

// For completeness, let's also check default trait bounds on `Self`.

trait C0 = std::marker::MetaSized; // has a user-written `Self: MetaSized` bound
fn h0<T: std::marker::PointeeSized>() where T: C0 {}

trait C1 =; // has a default `Self: MetaSized` bound
fn h1<T: std::marker::PointeeSized>() where T: C1 {}

fn main() {}
