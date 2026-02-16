// Check that trait alias bounds also imply default trait bounds to ensure that
// default and user-written bounds behave the same wrt. trait solving.
//
// Previously, we would only imply default trait bounds on
// *`Self`* type params, not on regular type params however.
// issue: <https://github.com/rust-lang/rust/issues/152687>

//@ compile-flags: -Zexperimental-default-bounds
//@ check-pass

#![feature(trait_alias, more_maybe_bounds, lang_items, auto_traits)]

#[lang = "default_trait1"]
auto trait Mark {}

trait A0<T: Mark> = ?Mark; // has a user-written `T: Mark` bound
fn f0<T: ?Mark>() where (): A0<T> {}

trait A1<T> = ?Mark; // has a default `T: Mark` bound
fn f1<T: ?Mark>() where (): A1<T> {}

// For completeness, let's also check default trait bounds on `Self`.

trait B0 = Mark; // has a user-written `Self: Mark` bound
fn g0<T: ?Mark>() where T: B0 { ensure::<T>; }

trait B1 =; // has a default `Self: Mark` bound
fn g1<T: ?Mark>() where T: B1 { ensure::<T>; }

fn ensure<T/*: Mark*/>() {}

fn main() {}
