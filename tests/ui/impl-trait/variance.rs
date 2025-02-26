//@ revisions: old e2024
//@[e2024] edition: 2024

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn not_captured_early<'a: 'a>() -> impl Sized {}
//[old]~^ ['a: *]
//[e2024]~^^ ['a: *, 'a: o]

fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {} //~ ['a: *, 'a: o]

fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}
//[old]~^ []
//[e2024]~^^ ['a: o]

fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {} //~ ['a: o]

fn main() {}
