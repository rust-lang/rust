// Check that trait alias bounds also imply default trait bounds to ensure that
// default and user-written bounds behave the same wrt. trait solving.
//
// Previously, we would only imply default trait bounds on
// *`Self`* type params, not on regular type params however.
// issue: <https://github.com/rust-lang/rust/issues/152687>
//
//@ revisions: pass fail
//@[pass] check-pass

#![feature(trait_alias)]
#![feature(sized_hierarchy, extern_types)] // only used for the `MetaSized` test cases

//------------------------- `Sized`

trait A0<T: Sized> =; // has a user-written `T: Sized` boun

// `(): A0<T>` requires+implies `T: Sized`
fn a0<T: ?Sized>() where (): A0<T> { ensure_is_sized::<T>; }


trait A1<T> =; // has a default `T: Sized` bound

// `(): A1<T>` requires+implies `T: Sized`
fn a1<T: ?Sized>() where (): A1<T> { ensure_is_sized::<T>; }
#[cfg(fail)] fn a1_fail() { a1::<str>; }
//[fail]~^ ERROR the size for values of type `str` cannot be known at compilation time


fn ensure_is_sized<T: Sized>() {}

//------------------------- `MetaSized`

use std::marker::{MetaSized, PointeeSized};
unsafe extern "C" { type ExternTy; }


trait B0<T: MetaSized> =; // has a user-written `T: MetaSized` bounds

// `(): B0<T>` requires+implies `T: MetaSized`
fn b0<T: PointeeSized>() where (): B0<T> { ensure_is_meta_sized::<T>(); }


trait B1<T: ?Sized> =; // has a default `T: MetaSized` bound

// `(): B1<T>` requires+implies `T: MetaSized`
fn b1<T: PointeeSized>() where (): B1<T> { ensure_is_meta_sized::<T>(); }

#[cfg(fail)] fn b1_fail() { b1::<ExternTy>; }
//[fail]~^ ERROR the size for values of type `ExternTy` cannot be known

// For completeness, let's also check default trait bounds on `Self`.

trait C0 = MetaSized; // has a user-written `Self: MetaSized` bound

// `T: C0` requires+implies `T: MetaSized`
fn c0<T: PointeeSized>() where T: C0 { ensure_is_meta_sized::<T>; }


trait C1 =; // has a default `Self: MetaSized` bound

// `T: C1` requires+implies `T: MetaSized`
fn c1<T: PointeeSized>() where T: C1 { ensure_is_meta_sized::<T>; }

#[cfg(fail)] fn c1_fail() { c1::<ExternTy>; }
//[fail]~^ ERROR the trait bound `ExternTy: C1` is not satisfied


fn ensure_is_meta_sized<T: MetaSized>() {}

fn main() {}
