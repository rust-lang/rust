//@ aux-build:pretty-print-dep.rs
//@ compile-flags: --crate-type=lib

extern crate pretty_print_dep;
use pretty_print_dep::{SizedTr, NegSizedTr, MetaSizedTr, PointeeSizedTr};

// Test that printing the sizedness trait bounds in the conflicting impl error without enabling
// `sized_hierarchy` will continue to print `?Sized`, even if the dependency is compiled with
// `sized_hierarchy`.
//
// It isn't possible to write a test that matches the multiline note containing the important
// diagnostic output being tested - so check the stderr changes carefully!

struct X<T>(T);

impl<T: Sized> SizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `SizedTr` for type `X<_>`

impl<T: ?Sized> NegSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `NegSizedTr` for type `X<_>`

impl<T: ?Sized> MetaSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `MetaSizedTr` for type `X<_>`

impl<T: ?Sized> PointeeSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `PointeeSizedTr` for type `X<_>`
