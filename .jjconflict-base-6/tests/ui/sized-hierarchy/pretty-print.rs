//@ aux-build:pretty-print-dep.rs
//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

// Test that printing the sizedness trait bounds in the conflicting impl error with
// `sized_hierarchy` enabled prints all of the appropriate bounds.
//
// It isn't possible to write a test that matches the multiline note containing the important
// diagnostic output being tested - so check the stderr changes carefully!

use std::marker::{MetaSized, PointeeSized};

extern crate pretty_print_dep;
use pretty_print_dep::{SizedTr, MetaSizedTr, PointeeSizedTr};

struct X<T>(T);

impl<T: Sized> SizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `SizedTr` for type `X<_>`

impl<T: ?Sized> pretty_print_dep::NegSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `NegSizedTr` for type `X<_>`

impl<T: MetaSized> MetaSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `MetaSizedTr` for type `X<_>`

impl<T: PointeeSized> PointeeSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `PointeeSizedTr` for type `X<_>`
