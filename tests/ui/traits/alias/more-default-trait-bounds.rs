// Check that trait alias bounds also imply default trait bounds to ensure_has_mark that
// default and user-written bounds behave the same wrt. trait solving.
//
// Previously, we would only imply default trait bounds on
// *`Self`* type params, not on regular type params however.
// issue: <https://github.com/rust-lang/rust/issues/152687>
//
//@ revisions: pass fail
//@ compile-flags: -Zexperimental-default-bounds
//@[pass] check-pass

#![feature(trait_alias, more_maybe_bounds, lang_items, auto_traits, negative_impls)]

#[lang = "default_trait1"]
auto trait Mark {}


trait A0<T: Mark> = ?Mark; // has a user-written `T: Mark` bound

fn a0<T: ?Mark>() where (): A0<T> { ensure_has_mark::<T>; }


trait A1<T> = ?Mark; // has a default `T: Mark` bound

fn a1<T: ?Mark>() where (): A1<T> { ensure_has_mark::<T>; }

#[cfg(fail)] fn a1_fail() { a1::<Unmarked>; }
//[fail]~^ ERROR the trait bound `Unmarked: Mark` is not satisfied

// For completeness, let's also check default trait bounds on `Self`.

trait B0 = Mark; // has a user-written `Self: Mark` bound

fn b0<T: ?Mark>() where T: B0 { ensure_has_mark::<T>; }


trait B1 =; // has a default `Self: Mark` bound

fn b1<T: ?Mark>() where T: B1 { ensure_has_mark::<T>; }

#[cfg(fail)] fn b1_fail() { b1::<Unmarked>; }
//[fail]~^ ERROR the trait bound `Unmarked: B1` is not satisfied


fn ensure_has_mark<T: Mark>() {}

enum Unmarked {}
impl !Mark for Unmarked {}

fn main() {}
