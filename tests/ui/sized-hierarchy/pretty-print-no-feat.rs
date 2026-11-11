//@ aux-build:pretty-print-no-feat-dep.rs
//@ compile-flags: --crate-type=lib

extern crate pretty_print_no_feat_dep;
use pretty_print_no_feat_dep::{SizedTr, NegSizedTr};

// Test that printing the sizedness trait bounds in the conflicting impl error without enabling
// `sized_hierarchy` will continue to print `?Sized`.
//
// It isn't possible to write a test that matches the multiline note containing the important
// diagnostic output being tested - so check the stderr changes carefully!

struct X<T>(T);

impl<T: Sized> SizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `SizedTr` for type `X<_>`

impl<T: ?Sized> NegSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `NegSizedTr` for type `X<_>`
