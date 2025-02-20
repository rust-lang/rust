//@ aux-build:pretty-print-dep.rs
//@ compile-flags: --crate-type=lib
#![feature(const_trait_impl, sized_hierarchy)]

// Test that printing the sizedness trait bounds in the conflicting impl error with
// `sized_hierarchy` enabled prints all of the appropriate bounds.
//
// It isn't possible to write a test that matches the multiline note containing the important
// diagnostic output being tested - so check the stderr changes carefully!

use std::marker::{MetaSized, PointeeSized};

extern crate pretty_print_dep;
use pretty_print_dep::{ConstSizedTr, SizedTr, ConstMetaSizedTr, MetaSizedTr, PointeeSizedTr};

struct X<T>(T);

impl<T: const Sized> ConstSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `ConstSizedTr` for type `X<_>`

impl<T: Sized> SizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `SizedTr` for type `X<_>`

impl<T: ?Sized> pretty_print_dep::NegSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `NegSizedTr` for type `X<_>`

impl<T: const MetaSized> ConstMetaSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `ConstMetaSizedTr` for type `X<_>`

impl<T: MetaSized> MetaSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `MetaSizedTr` for type `X<_>`

impl<T: PointeeSized> PointeeSizedTr for X<T> {}
//~^ ERROR conflicting implementations of trait `PointeeSizedTr` for type `X<_>`
